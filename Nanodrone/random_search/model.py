from __future__ import annotations

import torch
import torch.nn as nn


class _FF(nn.Module):
    """Simple feedforward network used to initialize the LSTM hidden state from y0."""
    def __init__(self, input_size, hidden_sizes, output_size, activation="ReLU", dropout_prob=0.0):
        super().__init__()
        act = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}.get(activation, nn.ReLU)
        layers: list[nn.Module] = []
        sizes = [input_size] + list(hidden_sizes)
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(act())
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(sizes[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _init_hidden(ff: _FF, y0: torch.Tensor, num_layers: int):
    """Map y0 -> (h0, c0) for LSTM, repeated across layers."""
    y0_flat = y0.squeeze(1) if y0.dim() == 3 else y0
    h = ff(y0_flat).unsqueeze(0).repeat(num_layers, 1, 1)
    return h, h.clone()


class AutoregressiveLSTM(nn.Module):
    """
    Black-box LSTM.
    - Hidden state initialized from y0 via a feedforward network.
    - At each step t the LSTM receives [u_t, y_{t-1}] as input.
    - Output at each step is y_t (full state).
    """
    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_hidden_states: int,
        hidden_sizes: list[int],
        num_layers: int,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.ff_init = _FF(n_states, hidden_sizes, n_hidden_states, activation, dropout_prob)
        self.lstm = nn.LSTM(
            input_size=n_inputs + n_outputs,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(n_hidden_states, n_outputs)

    def forward(self, u: torch.Tensor, y0: torch.Tensor, **_) -> tuple[torch.Tensor, object]:
        hidden = _init_hidden(self.ff_init, y0, self.num_layers)
        y0_flat = y0.squeeze(1) if y0.dim() == 3 else y0
        y_prev = y0_flat
        outputs = []
        for t in range(u.shape[1]):
            inp = torch.cat([u[:, t:t+1, :], y_prev.unsqueeze(1)], dim=-1)
            h, hidden = self.lstm(inp, hidden)
            y_t = self.out(h.squeeze(1))
            outputs.append(y_t.unsqueeze(1))
            y_prev = y_t
        return torch.cat(outputs, dim=1), hidden


class PhysicsResidualLSTM(nn.Module):
    """
    Hybrid model: kinematic physics step + LSTM residual.

    State layout (normalized): [x,y,z | vx,vy,vz | roll,pitch,yaw | wx,wy,wz]

    Physics step (Euler integration, gains are learnable):
        p_t     = p_{t-1}     + gain_p * v_{t-1}
        euler_t = euler_{t-1} + gain_e * omega_{t-1}
        v_t     = v_{t-1}       (zero-order hold)
        omega_t = omega_{t-1}   (zero-order hold)

    LSTM: [u_t, y_{t-1}] -> delta_t  (12-D residual correction)

    Output: y_t = y_phys_t + delta_t
    """
    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_hidden_states: int,
        hidden_sizes: list[int],
        num_layers: int,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.gain_p = nn.Parameter(torch.full((3,), 0.02))
        self.gain_e = nn.Parameter(torch.full((3,), 0.04))

        self.ff_init = _FF(n_states, hidden_sizes, n_hidden_states, activation, dropout_prob)
        self.lstm = nn.LSTM(
            input_size=n_inputs + n_outputs,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(n_hidden_states, n_outputs)

    def _physics_step(self, y: torch.Tensor) -> torch.Tensor:
        p, v, euler, omega = y[:, 0:3], y[:, 3:6], y[:, 6:9], y[:, 9:12]
        return torch.cat([
            p     + self.gain_p * v,
            v,
            euler + self.gain_e * omega,
            omega,
        ], dim=-1)

    def forward(self, u: torch.Tensor, y0: torch.Tensor, **_) -> tuple[torch.Tensor, object]:
        hidden = _init_hidden(self.ff_init, y0, self.num_layers)
        y0_flat = y0.squeeze(1) if y0.dim() == 3 else y0
        y_prev = y0_flat
        outputs = []
        for t in range(u.shape[1]):
            y_phys = self._physics_step(y_prev)
            inp = torch.cat([u[:, t:t+1, :], y_prev.unsqueeze(1)], dim=-1)
            h, hidden = self.lstm(inp, hidden)
            delta = self.out(h.squeeze(1))
            y_t = y_phys + delta
            outputs.append(y_t.unsqueeze(1))
            y_prev = y_t
        return torch.cat(outputs, dim=1), hidden


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int):
    cls = config_pars["model_class"]
    kwargs = dict(
        n_inputs=n_inputs,
        n_states=n_states,
        n_outputs=n_outputs,
        n_hidden_states=config_pars["n_hidden_states"],
        hidden_sizes=config_pars["hidden_sizes"],
        num_layers=config_pars["num_layers"],
        activation=config_pars.get("activation", "ReLU"),
        dropout_prob=config_pars.get("dropout_prob", 0.0),
    )
    if cls == "AutoregressiveLSTM":
        return AutoregressiveLSTM(**kwargs)
    if cls == "PhysicsResidualLSTM":
        return PhysicsResidualLSTM(**kwargs)
    raise ValueError(f"Unknown model_class: {cls!r}")
