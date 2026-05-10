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
        h0_single = self.FF_initial(y0)                   # (batch, n_hidden)
        h0 = h0_single.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (layers, batch, hidden)
        if self.recurrent_type == "LSTM":
            return (h0, h0.clone())
        return h0

    def forward(
        self,
        u: torch.Tensor,   # (batch, seq_len, n_inputs)
        y0: torch.Tensor,  # (batch, 1, n_states) or (batch, n_states)
    ) -> tuple[torch.Tensor, object]:
        init_state = self.build_initial_state(y0)
        y_hat, hidden_state = self.recurrence(u, init_state)
        return y_hat, hidden_state


def build_model_from_config(
    config_pars: dict,
    n_inputs: int,
    n_states: int,
    n_outputs: int,
) -> MyDroneModel:
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
