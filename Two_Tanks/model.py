from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    warnings.warn(f"Unknown activation `{name}`. Falling back to ReLU.")
    return nn.ReLU()


class FeedforwardNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one element")

        layers: list[nn.Module] = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(make_activation(activation))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        recurrent: str = "LSTM",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        recurrent_key = recurrent.upper()
        common_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout_prob if num_layers > 1 else 0.0,
        }

        if recurrent_key == "LSTM":
            self.recurrent = nn.LSTM(**common_args)
        elif recurrent_key == "GRU":
            self.recurrent = nn.GRU(**common_args)
        elif recurrent_key == "RNN":
            self.recurrent = nn.RNN(**common_args)
        else:
            raise ValueError("recurrent must be one of {'RNN', 'GRU', 'LSTM'}")

        self.recurrent_type = recurrent_key
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        output, hidden_state = self.recurrent(x, hidden_state)
        y = self.output_layer(output)
        return y, hidden_state


class CascadedTankModel(nn.Module):
    def __init__(
        self,
        n_inputs: int = 1,
        n_states: int = 1,
        n_outputs: int = 1,
        n_hidden_states: int = 32,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 1,
        recurrent: str = "LSTM",
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
        direct_feedthrough: bool = False,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32]

        self.n_inputs = n_inputs
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent.upper()
        self.direct_feedthrough = direct_feedthrough

        self.initial_state_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.recurrent_block = RecurrentBlock(
            input_size=n_inputs,
            hidden_size=n_hidden_states,
            output_size=n_outputs,
            num_layers=num_layers,
            recurrent=recurrent,
            dropout_prob=dropout_prob,
        )

        self.direct_layer = None
        if direct_feedthrough:
            self.direct_layer = nn.Linear(n_inputs, n_outputs)

    def build_initial_state(
        self, y0: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        if y0.dim() != 2:
            raise ValueError("y0 must have shape (batch, n_states) or (batch, 1, n_states)")

        hidden_single = self.initial_state_net(y0)
        hidden = hidden_single.unsqueeze(0).repeat(self.num_layers, 1, 1)

        if self.recurrent_type == "LSTM":
            cell = hidden.clone()
            return hidden, cell
        return hidden

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, object]:
        initial_state = self.build_initial_state(y0)
        y_hat, hidden_state = self.recurrent_block(u, initial_state)
        if self.direct_layer is not None:
            y_hat = y_hat + self.direct_layer(u)
        return y_hat, hidden_state


class PhysicalTankModel(nn.Module):
    """
    Grey-box model for the cascaded two-tank system.
    Implements discrete-time Euler integration of Torricelli's ODE:
      dz1/dt = -k1*sqrt(z1+) + k4*u
      dz2/dt =  k2*sqrt(z1+) - k3*sqrt(z2+)
      y = out_scale * z2 + out_bias
    All states and signals are in normalized space.
    k1-k4 and dt are positive (log-parameterized).
    Initial [z1, z2] are predicted from y0 by a small MLP.
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        self.state_init = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=2,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.log_k1 = nn.Parameter(torch.zeros(1))
        self.log_k2 = nn.Parameter(torch.zeros(1))
        self.log_k3 = nn.Parameter(torch.zeros(1))
        self.log_k4 = nn.Parameter(torch.zeros(1))
        self.log_dt = nn.Parameter(torch.zeros(1))

        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, u: torch.Tensor, y0: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape

        z_init = self.state_init(y0)
        z1 = z_init[:, 0:1]
        z2 = z_init[:, 1:2]

        k1 = torch.exp(self.log_k1)
        k2 = torch.exp(self.log_k2)
        k3 = torch.exp(self.log_k3)
        k4 = torch.exp(self.log_k4)
        dt = torch.exp(self.log_dt)

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            sqrt_z1 = torch.sqrt(F.softplus(z1))
            sqrt_z2 = torch.sqrt(F.softplus(z2))
            dz1 = -k1 * sqrt_z1 + k4 * u_t
            dz2 = k2 * sqrt_z1 - k3 * sqrt_z2
            z1 = z1 + dt * dz1
            z2 = z2 + dt * dz2
            outputs.append(self.out_scale * z2 + self.out_bias)

        y_hat = torch.stack(outputs, dim=1)
        return y_hat, (z1, z2)


class PhysicalTankModelRK4(nn.Module):
    """
    Grey-box model with 4th-order Runge-Kutta integration (more accurate than Euler).
    Also adds learnable overflow nonlinearity via softmax gate.
    dx1/dt = -k1*sqrt(x1+) + k4*u - k5*overflow(x1)
    dx2/dt =  k2*sqrt(x1+) - k3*sqrt(x2+) + k6*overflow(x1)
    overflow(x1) = softplus(x1 - x1_max)
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        self.state_init = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=2,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.log_k1 = nn.Parameter(torch.zeros(1))
        self.log_k2 = nn.Parameter(torch.zeros(1))
        self.log_k3 = nn.Parameter(torch.zeros(1))
        self.log_k4 = nn.Parameter(torch.zeros(1))
        self.log_k5 = nn.Parameter(torch.full((1,), -2.0))
        self.log_k6 = nn.Parameter(torch.full((1,), -2.0))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.x1_max = nn.Parameter(torch.tensor(2.0))

        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def _ode(self, z1, z2, u_t, k1, k2, k3, k4, k5, k6):
        sqrt_z1 = torch.sqrt(F.softplus(z1))
        sqrt_z2 = torch.sqrt(F.softplus(z2))
        overflow = F.softplus(z1 - self.x1_max)
        dz1 = -k1 * sqrt_z1 + k4 * u_t - k5 * overflow
        dz2 = k2 * sqrt_z1 - k3 * sqrt_z2 + k6 * overflow
        return dz1, dz2

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        z_init = self.state_init(y0)
        z1 = z_init[:, 0:1]
        z2 = z_init[:, 1:2]

        k1 = torch.exp(self.log_k1)
        k2 = torch.exp(self.log_k2)
        k3 = torch.exp(self.log_k3)
        k4 = torch.exp(self.log_k4)
        k5 = torch.exp(self.log_k5)
        k6 = torch.exp(self.log_k6)
        dt = torch.exp(self.log_dt)

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            k1a, k1b = self._ode(z1, z2, u_t, k1, k2, k3, k4, k5, k6)
            k2a, k2b = self._ode(z1 + 0.5 * dt * k1a, z2 + 0.5 * dt * k1b, u_t, k1, k2, k3, k4, k5, k6)
            k3a, k3b = self._ode(z1 + 0.5 * dt * k2a, z2 + 0.5 * dt * k2b, u_t, k1, k2, k3, k4, k5, k6)
            k4a, k4b = self._ode(z1 + dt * k3a, z2 + dt * k3b, u_t, k1, k2, k3, k4, k5, k6)
            z1 = z1 + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
            z2 = z2 + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
            outputs.append(self.out_scale * z2 + self.out_bias)

        y_hat = torch.stack(outputs, dim=1)
        return y_hat, (z1, z2)


class HybridTankModel(nn.Module):
    """
    Hybrid: PhysicalTankModel backbone + LSTM residual correction.
    The physical model provides a coarse prediction; the LSTM corrects
    unmodeled effects (overflow, measurement noise, etc.).
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_hidden_states: int = 16,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 1,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        self.physical = PhysicalTankModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.residual_block = RecurrentBlock(
            input_size=n_inputs,
            hidden_size=n_hidden_states,
            output_size=n_outputs,
            num_layers=num_layers,
            recurrent="LSTM",
            dropout_prob=dropout_prob,
        )

        self.residual_init_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, u: torch.Tensor, y0: torch.Tensor
    ) -> tuple[torch.Tensor, object]:
        y_phys, _ = self.physical(u, y0)

        if y0.dim() == 3:
            y0_flat = y0.squeeze(1)
        else:
            y0_flat = y0
        h0 = self.residual_init_net(y0_flat).unsqueeze(0).repeat(
            self.residual_block.recurrent.num_layers, 1, 1
        )
        c0 = h0.clone()
        y_res, hidden = self.residual_block(u, (h0, c0))
        y_hat = y_phys + self.residual_scale * y_res
        return y_hat, hidden


class ClosedFormCTC(nn.Module):
    """
    Closed-form Continuous-time network (CfC).
    Each neuron has a learnable time constant tau_i > 0.
    State update: h[t+1] = A_i * h[t] + (1-A_i) * f(h[t], u[t])
    where A_i = exp(-dt/tau_i) in (0,1) — unconditional stability.
    dt is the known sampling time (seconds).
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_hidden_states: int = 64,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        direct_feedthrough: bool = False,
        dt: float = 4.0,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.n_hidden = n_hidden_states
        self.direct_feedthrough = direct_feedthrough

        # Initialize tau so neurons span [dt, 100s] in log space.
        log_tau_init = torch.linspace(
            torch.log(torch.tensor(dt)),
            torch.log(torch.tensor(100.0)),
            n_hidden_states,
        )
        self.log_tau = nn.Parameter(log_tau_init)
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float32))

        # Dynamics MLP: [h, u] -> target state
        self.f_net = FeedforwardNetwork(
            input_size=n_hidden_states + n_inputs,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        # IC network: y0 -> initial h
        self.ic_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        readout_in = n_hidden_states + n_inputs if direct_feedthrough else n_hidden_states
        self.readout = nn.Linear(readout_in, n_outputs)

    def forward(
        self, u: torch.Tensor, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch, T, _ = u.shape
        h = self.ic_net(y0)

        tau = torch.exp(self.log_tau)
        A = torch.exp(-self.dt / tau).unsqueeze(0)  # (1, n_hidden)

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            h = A * h + (1.0 - A) * self.f_net(torch.cat([h, u_t], dim=-1))
            if self.direct_feedthrough:
                y_t = self.readout(torch.cat([h, u_t], dim=-1))
            else:
                y_t = self.readout(h)
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h


def build_model_from_config(
    config_pars: dict,
    n_inputs: int,
    n_states: int,
    n_outputs: int,
) -> nn.Module:
    model_type = str(config_pars.get("type", "RNN")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return CascadedTankModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"],
            recurrent=model_type,
            activation=activation,
            dropout_prob=dropout_prob,
            direct_feedthrough=config_pars.get("direct_feedthrough", False),
        )

    if model_type == "LTC":
        return ClosedFormCTC(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            direct_feedthrough=config_pars.get("direct_feedthrough", False),
            dt=float(config_pars.get("dt", 4.0)),
        )

    if model_type == "PHYSICAL_RK4":
        return PhysicalTankModelRK4(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
        )

    if model_type == "PHYSICAL":
        return PhysicalTankModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
        )

    if model_type == "HYBRID":
        return HybridTankModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars.get("n_hidden_states", 16),
            hidden_sizes=hidden_sizes,
            num_layers=config_pars.get("num_layers", 1),
            activation=activation,
            dropout_prob=dropout_prob,
        )

    raise ValueError(
        "Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'LTC', 'PHYSICAL', 'HYBRID'}."
    )
