from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="ReLU", dropout_prob=0.0):
        super().__init__()

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one element")

        if activation == "ReLU":
            act = nn.ReLU
        elif activation == "Tanh":
            act = nn.Tanh
        elif activation == "sigmoid":
            act = nn.Sigmoid
        else:
            warnings.warn(f"Activation '{activation}' not recognised — using ReLU.")
            act = nn.ReLU

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(act())
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(act())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MyRecurrence(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, recurrent="LSTM"):
        super().__init__()

        self.recurrent_type = recurrent

        if recurrent == "LSTM":
            self.recurrence = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif recurrent == "GRU":
            self.recurrence = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif recurrent == "RNN":
            self.recurrence = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"recurrent must be 'LSTM', 'GRU', or 'RNN', got '{recurrent}'")

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        output, hidden_state = self.recurrence(x, hidden_state)
        return self.output_layer(output), hidden_state


class MyDroneModel(nn.Module):
    def __init__(
        self,
        n_inputs: int = 4,
        n_states: int = 12,
        n_outputs: int = 12,
        n_hidden_states: int = 32,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 1,
        recurrent: str = "LSTM",
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32]

        self.n_inputs = n_inputs
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent

        self.FF_initial = FeedforwardNeuralNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.recurrence = MyRecurrence(
            input_size=n_inputs,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
            output_size=n_outputs,
            recurrent=recurrent,
        )

    def build_initial_state(self, y0: torch.Tensor):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        h0_single = self.FF_initial(y0)
        h0 = h0_single.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.recurrent_type == "LSTM":
            return (h0, h0.clone())
        return h0

    def forward(
        self,
        u: torch.Tensor,
        y0: torch.Tensor,
        y_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object]:
        init_state = self.build_initial_state(y0)
        y_hat, hidden_state = self.recurrence(u, init_state)
        return y_hat, hidden_state


class AutoregressiveDeltaGRU(nn.Module):
    """
    Causal autoregressive GRU with delta-state prediction.
    Input per step: [u_t, u_t^2, y_{t-1}]  (motor speeds + squares + prev state)
    Output: delta_t = y_t - y_{t-1}
    y_t = y_{t-1} + delta_t
    Hidden state initialised from y0 via a feedforward network.
    Teacher forcing: during training, y_{t-1} can be the ground truth.
    """

    def __init__(
        self,
        n_inputs: int = 4,
        n_states: int = 12,
        n_outputs: int = 12,
        n_hidden_states: int = 256,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 3,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.n_outputs = n_outputs
        self.num_layers = num_layers

        self.ff_init = FeedforwardNeuralNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation="ReLU",
            dropout_prob=dropout_prob,
        )

        # input: [u_t (4), u_t^2 (4), y_{t-1} (12)] = 20
        self.gru = nn.GRU(
            input_size=n_inputs * 2 + n_states,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(n_hidden_states, n_outputs)

    def _init_hidden(self, y0: torch.Tensor) -> torch.Tensor:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        h0 = self.ff_init(y0)
        return h0.unsqueeze(0).repeat(self.num_layers, 1, 1)

    def forward(
        self,
        u: torch.Tensor,          # (B, T, 4)
        y0: torch.Tensor,         # (B, 1, 12) or (B, 12)
        y_teacher: torch.Tensor | None = None,  # (B, T, 12) ground truth for teacher forcing
    ) -> tuple[torch.Tensor, object]:
        B, T, _ = u.shape
        h = self._init_hidden(y0)

        if y0.dim() == 3:
            y_prev = y0.squeeze(1)
        else:
            y_prev = y0

        outputs = []
        for t in range(T):
            ut = u[:, t, :]
            inp = torch.cat([ut, ut ** 2, y_prev], dim=-1).unsqueeze(1)
            out, h = self.gru(inp, h)
            delta = self.out(out.squeeze(1))
            y_t = y_prev + delta
            outputs.append(y_t.unsqueeze(1))
            if y_teacher is not None:
                y_prev = y_teacher[:, t, :]
            else:
                y_prev = y_t

        return torch.cat(outputs, dim=1), h


class PhysicsResidualCausal(nn.Module):
    """
    Causal physics-structured residual model.

    State ordering: [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]  (indices 0-11)

    Physics prior (causal, normalized space):
      Δpos   = diag(γ_p) * vel_{t-1}                             (kinematic)
      Δvel   = thrust_net([u², sin/cos(euler)_{t-1}])            (motor thrust)
      Δeuler = diag(γ_e) * omega_{t-1}                           (kinematic)
      Δomega = motor_mix(u²) + diag(γ_d) * omega_{t-1}          (motor torques + damping)

    GRU residual corrects remaining errors:
      Δy_res = GRU([u, u², y_{t-1}], h_t)

    y_t = y_{t-1} + Δy_physics + Δy_res
    """

    # State slice indices
    POS   = slice(0, 3)
    VEL   = slice(3, 6)
    EULER = slice(6, 9)
    OMEGA = slice(9, 12)

    def __init__(
        self,
        n_inputs: int = 4,
        n_states: int = 12,
        n_outputs: int = 12,
        n_hidden_states: int = 256,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 3,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.n_outputs = n_outputs
        self.num_layers = num_layers

        # --- Physics parameters ---
        # Kinematic gains: pos integrates vel, euler integrates omega
        self.gamma_p = nn.Parameter(torch.ones(3) * 0.01)
        self.gamma_e = nn.Parameter(torch.ones(3) * 0.01)
        # Damping on omega
        self.gamma_d = nn.Parameter(torch.ones(3) * (-0.01))

        # Thrust: [u1², u2², u3², u4², sin(r), cos(r), sin(p), cos(p)] → Δvel (3)
        # 2-layer MLP for nonlinear thrust mapping (linear was too restrictive)
        self.thrust_net = nn.Sequential(
            nn.Linear(8, 32, bias=True),
            nn.Tanh(),
            nn.Linear(32, 3, bias=True),
        )
        nn.init.zeros_(self.thrust_net[0].weight)
        nn.init.zeros_(self.thrust_net[0].bias)
        nn.init.zeros_(self.thrust_net[2].weight)
        nn.init.zeros_(self.thrust_net[2].bias)

        # Motor mixing: [u1², u2², u3², u4²] → Δomega (3)
        # 2-layer MLP for nonlinear torque mapping
        self.motor_mix = nn.Sequential(
            nn.Linear(4, 16, bias=True),
            nn.Tanh(),
            nn.Linear(16, 3, bias=False),
        )
        nn.init.zeros_(self.motor_mix[0].weight)
        nn.init.zeros_(self.motor_mix[0].bias)
        nn.init.zeros_(self.motor_mix[2].weight)

        # --- GRU residual ---
        self.ff_init = FeedforwardNeuralNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation="ReLU",
            dropout_prob=dropout_prob,
        )
        self.gru = nn.GRU(
            input_size=n_inputs * 2 + n_states,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )
        self.res_out = nn.Linear(n_hidden_states, n_outputs)

    def _init_hidden(self, y0: torch.Tensor) -> torch.Tensor:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        h0 = self.ff_init(y0)
        return h0.unsqueeze(0).repeat(self.num_layers, 1, 1)

    def _physics_step(self, y_prev: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One-step causal physics delta. y_prev: (B,12), u: (B,4)."""
        u2 = u ** 2
        roll  = y_prev[:, 6]
        pitch = y_prev[:, 7]

        # Kinematics
        delta_pos   = self.gamma_p * y_prev[:, self.VEL]
        delta_euler = self.gamma_e * y_prev[:, self.OMEGA]

        # Thrust from motors + attitude
        thrust_feat = torch.cat([
            u2,
            torch.sin(roll).unsqueeze(1), torch.cos(roll).unsqueeze(1),
            torch.sin(pitch).unsqueeze(1), torch.cos(pitch).unsqueeze(1),
        ], dim=-1)
        delta_vel = self.thrust_net(thrust_feat)

        # Motor torques + damping
        delta_omega = self.motor_mix(u2) + self.gamma_d * y_prev[:, self.OMEGA]

        return torch.cat([delta_pos, delta_vel, delta_euler, delta_omega], dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        y0: torch.Tensor,
        y_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object]:
        B, T, _ = u.shape
        h = self._init_hidden(y0)

        if y0.dim() == 3:
            y_prev = y0.squeeze(1)
        else:
            y_prev = y0

        outputs = []
        for t in range(T):
            ut = u[:, t, :]
            delta_phys = self._physics_step(y_prev, ut)

            inp = torch.cat([ut, ut ** 2, y_prev], dim=-1).unsqueeze(1)
            out, h = self.gru(inp, h)
            delta_res = self.res_out(out.squeeze(1))

            y_t = y_prev + delta_phys + delta_res
            outputs.append(y_t.unsqueeze(1))

            if y_teacher is not None:
                y_prev = y_teacher[:, t, :]
            else:
                y_prev = y_t

        return torch.cat(outputs, dim=1), h


def build_model_from_config(
    config_pars: dict,
    n_inputs: int,
    n_states: int,
    n_outputs: int,
) -> nn.Module:
    model_type = config_pars.get("type", "LSTM")

    if model_type == "PhysicsResidualCausal":
        return PhysicsResidualCausal(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=config_pars.get("hidden_sizes", [128, 64]),
            num_layers=config_pars.get("num_layers", 3),
            dropout_prob=config_pars.get("dropout_prob", 0.1),
        )

    if model_type == "AutoregressiveDeltaGRU":
        return AutoregressiveDeltaGRU(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=config_pars.get("hidden_sizes", [128, 64]),
            num_layers=config_pars.get("num_layers", 3),
            dropout_prob=config_pars.get("dropout_prob", 0.1),
        )

    return MyDroneModel(
        n_inputs=n_inputs,
        n_states=n_states,
        n_outputs=n_outputs,
        n_hidden_states=config_pars["n_hidden_states"],
        hidden_sizes=config_pars["hidden_sizes"],
        num_layers=config_pars["num_layers"],
        recurrent=model_type,
        activation=config_pars["activation"],
        dropout_prob=config_pars.get("dropout_prob", 0.0),
    )
