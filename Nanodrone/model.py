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
    ) -> tuple[torch.Tensor, object]:
        init_state = self.build_initial_state(y0)
        y_hat, hidden_state = self.recurrence(u, init_state)
        return y_hat, hidden_state


class KinematicsLSTMModel(nn.Module):
    """
    Physics-inspired kinematic model with motor-squared inputs.

    Physical structure:
    - Position/Euler integrated from velocities: p_t = p_{t-1} + gain_p * v_{t-1}
    - Euler_t = euler_{t-1} + gain_e * omega_{t-1}
    - LSTM predicts delta_v and delta_omega (residual dynamics)
    - Motor input augmented with squared terms [u, u^2] (8D) to expose thrust signal

    State layout: [x,y,z (0:3), vx,vy,vz (3:6), roll,pitch,yaw (6:9), wx,wy,wz (9:12)]
    """

    def __init__(
        self,
        n_inputs: int = 4,
        n_states: int = 12,
        n_outputs: int = 12,
        n_hidden_states: int = 192,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 2,
        recurrent: str = "LSTM",
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
        use_squared_inputs: bool = True,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.n_inputs = n_inputs
        self.n_states = n_states
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent
        self.use_squared_inputs = use_squared_inputs

        # Learnable kinematic gains: [p_x, p_y, p_z, euler_roll, euler_pitch, euler_yaw]
        self.kin_gain = nn.Parameter(torch.tensor([0.02, 0.02, 0.01, 0.05, 0.04, 0.04]))

        # Init network: y0 (12D) -> hidden state
        self.FF_initial = FeedforwardNeuralNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        # LSTM input: [u(4), u^2(4), full_state(12)] = 20D  or [u(4), u^2(4), v(3), omega(3)] = 14D
        # Use full state input; if use_squared_inputs: n_inputs*2 + n_states else n_inputs + n_states
        motor_dim = n_inputs * 2 if use_squared_inputs else n_inputs
        lstm_in = motor_dim + n_states

        self.recurrence = MyRecurrence(
            input_size=lstm_in,
            hidden_size=n_hidden_states,
            num_layers=num_layers,
            output_size=6,  # delta_v (3) + delta_omega (3)
            recurrent=recurrent,
        )

    def build_initial_state(self, y0: torch.Tensor):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        h0 = self.FF_initial(y0).unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.recurrent_type == "LSTM":
            return (h0, h0.clone())
        return h0

    def forward(
        self,
        u: torch.Tensor,
        y0: torch.Tensor,
        teacher_targets: torch.Tensor | None = None,
        teacher_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, object]:
        y0_flat = y0.squeeze(1) if y0.dim() == 3 else y0
        hidden  = self.build_initial_state(y0_flat)

        y_prev  = y0_flat
        outputs = []

        for t in range(u.shape[1]):
            # Scheduled sampling: replace y_prev with ground truth with prob teacher_ratio
            if t > 0 and teacher_targets is not None and teacher_ratio > 0.0:
                if torch.rand(1).item() < teacher_ratio:
                    y_prev = teacher_targets[:, t - 1, :]

            p_prev     = y_prev[:, 0:3]
            v_prev     = y_prev[:, 3:6]
            euler_prev = y_prev[:, 6:9]
            omega_prev = y_prev[:, 9:12]

            u_t = u[:, t, :]

            if self.use_squared_inputs:
                motor_feat = torch.cat([u_t, u_t ** 2], dim=-1)
            else:
                motor_feat = u_t

            lstm_in = torch.cat([motor_feat.unsqueeze(1), y_prev.unsqueeze(1)], dim=-1)
            dyn_out, hidden = self.recurrence(lstm_in, hidden)
            dyn_out = dyn_out.squeeze(1)

            v_t     = v_prev     + dyn_out[:, :3]
            omega_t = omega_prev + dyn_out[:, 3:]
            p_t     = p_prev     + self.kin_gain[:3] * v_prev
            euler_t = euler_prev + self.kin_gain[3:] * omega_prev

            y_t    = torch.cat([p_t, v_t, euler_t, omega_t], dim=-1)
            y_prev = y_t
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), hidden


def build_model_from_config(
    config_pars: dict,
    n_inputs: int,
    n_states: int,
    n_outputs: int,
):
    model_type = config_pars.get("model_class", "MyDroneModel")

    if model_type == "KinematicsLSTMModel":
        return KinematicsLSTMModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=config_pars["hidden_sizes"],
            num_layers=config_pars["num_layers"],
            recurrent=config_pars["type"],
            activation=config_pars["activation"],
            dropout_prob=config_pars.get("dropout_prob", 0.0),
            use_squared_inputs=config_pars.get("use_squared_inputs", True),
        )

    return MyDroneModel(
        n_inputs=n_inputs,
        n_states=n_states,
        n_outputs=n_outputs,
        n_hidden_states=config_pars["n_hidden_states"],
        hidden_sizes=config_pars["hidden_sizes"],
        num_layers=config_pars["num_layers"],
        recurrent=config_pars["type"],
        activation=config_pars["activation"],
        dropout_prob=config_pars.get("dropout_prob", 0.0),
    )
