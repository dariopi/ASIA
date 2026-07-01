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
    def __init__(self, input_size, hidden_sizes, output_size, activation="ReLU", dropout_prob=0.0):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one element")
        layers = []
        in_f = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_f, h), make_activation(activation)]
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            in_f = h
        layers.append(nn.Linear(in_f, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RecurrentBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, recurrent="RNN", dropout_prob=0.0):
        super().__init__()
        rk = recurrent.upper()
        args = dict(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout_prob if num_layers > 1 else 0.0)
        if rk == "RNN":
            self.recurrent = nn.RNN(**args)
        elif rk == "GRU":
            self.recurrent = nn.GRU(**args)
        elif rk == "LSTM":
            self.recurrent = nn.LSTM(**args)
        else:
            raise ValueError("recurrent must be RNN | GRU | LSTM")
        self.recurrent_type = rk
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.recurrent(x, h)
        return self.output_layer(out), h


# ──────────────────────────────────────────────
# 1. Classic recurrent (RNN / GRU / LSTM)
# ──────────────────────────────────────────────

class CascadedTankModel(nn.Module):
    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_hidden_states=4,
                 hidden_sizes=None, num_layers=1, recurrent="RNN",
                 activation="Tanh", dropout_prob=0.0, direct_feedthrough=False):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [4]
        self.recurrent_type = recurrent.upper()
        self.num_layers = num_layers
        self.direct_feedthrough = direct_feedthrough
        self.initial_state_net = FeedforwardNetwork(n_states, hidden_sizes, n_hidden_states, activation, dropout_prob)
        self.recurrent_block = RecurrentBlock(n_inputs, n_hidden_states, n_outputs, num_layers, recurrent, dropout_prob)
        self.direct_layer = nn.Linear(n_inputs, n_outputs) if direct_feedthrough else None

    def _init_h(self, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        h = self.initial_state_net(y0).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return (h, h.clone()) if self.recurrent_type == "LSTM" else h

    def get_initial_hidden(self, y0):
        return self._init_h(y0)

    def forward_from_hidden(self, u, h):
        y, h2 = self.recurrent_block(u, h)
        if self.direct_layer is not None:
            y = y + self.direct_layer(u)
        return y, h2

    def forward(self, u, y0):
        return self.forward_from_hidden(u, self._init_h(y0))


# ──────────────────────────────────────────────
# 2. Physics grey-box: RK4 Torricelli ODE
# ──────────────────────────────────────────────

class PhysicalRK4Model(nn.Module):
    def __init__(self, n_inputs, n_states, n_outputs, Ts=4.0, ic_hidden=32):
        super().__init__()
        self.Ts = Ts
        self.log_k = nn.Parameter(torch.zeros(4))
        self.ic_net = nn.Sequential(nn.Linear(n_states, ic_hidden), nn.Tanh(), nn.Linear(ic_hidden, 2))
        self.out = nn.Linear(1, n_outputs)

    def _f(self, x, u, k):
        x1, x2 = x[:, :1], x[:, 1:]
        dx1 = -k[0] * torch.sqrt(torch.clamp(x1, min=0)) + k[3] * u
        dx2 =  k[1] * torch.sqrt(torch.clamp(x1, min=0)) - k[2] * torch.sqrt(torch.clamp(x2, min=0))
        return torch.cat([dx1, dx2], -1)

    def forward(self, u, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        k = F.softplus(self.log_k)
        x = self.ic_net(y0)
        dt, outs = self.Ts, []
        for t in range(u.shape[1]):
            ut = u[:, t]
            k1 = self._f(x, ut, k)
            k2 = self._f(x + .5*dt*k1, ut, k)
            k3 = self._f(x + .5*dt*k2, ut, k)
            k4 = self._f(x + dt*k3, ut, k)
            x = torch.clamp(x + dt/6*(k1+2*k2+2*k3+k4), -10, 10)
            outs.append(self.out(x[:, 1:]))
        return torch.stack(outs, 1), x


# ──────────────────────────────────────────────
# 3. LTC / CfC (Closed-form Continuous-time)
# ──────────────────────────────────────────────

class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size, Ts=4.0, tau_min=4.0, tau_max=None, dropout_prob=0.0):
        super().__init__()
        self.Ts, self.tau_min, self.tau_max = Ts, tau_min, tau_max
        self.f_net   = nn.Linear(input_size + hidden_size, hidden_size)
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, u_t, h):
        xu = torch.cat([u_t, h], -1)
        f   = torch.tanh(self.f_net(xu))
        tau = F.softplus(self.tau_net(xu)) + self.tau_min
        if self.tau_max:
            tau = torch.clamp(tau, max=self.tau_max)
        return self.drop((h - f) * torch.exp(-self.Ts / tau) + f)


class LTCModel(nn.Module):
    def __init__(self, n_inputs, n_states, n_outputs, n_hidden=64,
                 Ts=4.0, tau_min=4.0, tau_max=None, dropout_prob=0.0):
        super().__init__()
        self.cell = LTCCell(n_inputs, n_hidden, Ts, tau_min, tau_max, dropout_prob)
        self.ic   = nn.Sequential(nn.Linear(n_states, n_hidden), nn.Tanh())
        self.out  = nn.Linear(n_hidden, n_outputs)

    def get_initial_hidden(self, y0):
        return self.ic(y0.squeeze(1) if y0.dim() == 3 else y0)

    def forward_from_hidden(self, u, h):
        outs = []
        for t in range(u.shape[1]):
            h = self.cell(u[:, t], h)
            outs.append(self.out(h))
        return torch.stack(outs, 1), h

    def forward(self, u, y0):
        return self.forward_from_hidden(u, self.get_initial_hidden(y0))


# ──────────────────────────────────────────────
# 4. Diagonal SSM  (S4-style, linear)
# ──────────────────────────────────────────────

class DiagonalSSMModel(nn.Module):
    def __init__(self, n_inputs, n_states, n_outputs, n_hidden=128):
        super().__init__()
        self.A_diag = nn.Parameter(torch.zeros(n_hidden))
        self.B = nn.Linear(n_inputs, n_hidden, bias=False)
        self.C = nn.Linear(n_hidden, n_outputs)
        self.D = nn.Linear(n_inputs, n_outputs)
        self.ic = nn.Sequential(nn.Linear(n_states, n_hidden), nn.Tanh())

    def get_initial_hidden(self, y0):
        return self.ic(y0.squeeze(1) if y0.dim() == 3 else y0)

    def forward_from_hidden(self, u, h):
        A = torch.tanh(self.A_diag)
        outs = []
        for t in range(u.shape[1]):
            ut = u[:, t]
            h = A * h + self.B(ut)
            outs.append(self.C(h) + self.D(ut))
        return torch.stack(outs, 1), h

    def forward(self, u, y0):
        return self.forward_from_hidden(u, self.get_initial_hidden(y0))


# ──────────────────────────────────────────────
# 5. Causal Dilated CNN (WaveNet-style)
# ──────────────────────────────────────────────

class CausalDilatedCNNModel(nn.Module):
    def __init__(self, n_inputs, n_states, n_outputs, n_channels=32, n_layers=7):
        super().__init__()
        self.ic_bias  = nn.Linear(n_states, n_channels)
        self.inp_proj = nn.Conv1d(n_inputs, n_channels, 1)
        dilations = [2**i for i in range(n_layers)]
        self.dilations = dilations
        self.conv_t = nn.ModuleList(nn.Conv1d(n_channels, n_channels, 2, dilation=d) for d in dilations)
        self.conv_s = nn.ModuleList(nn.Conv1d(n_channels, n_channels, 2, dilation=d) for d in dilations)
        self.res    = nn.ModuleList(nn.Conv1d(n_channels, n_channels, 1) for _ in dilations)
        self.out    = nn.Linear(n_channels, n_outputs)

    def forward(self, u, y0):
        y0f = y0.squeeze(1) if y0.dim() == 3 else y0
        x = self.inp_proj(u.transpose(1, 2)) + self.ic_bias(y0f).unsqueeze(-1)
        for ct, cs, res, d in zip(self.conv_t, self.conv_s, self.res, self.dilations):
            xp = F.pad(x, (d, 0))
            x  = x + res(torch.tanh(ct(xp)) * torch.sigmoid(cs(xp)))
        return self.out(x.transpose(1, 2)), None


# ──────────────────────────────────────────────
# 6. Hybrid LTC: physics Euler (detached) + LTC
# ──────────────────────────────────────────────

class HybridLTCModel(nn.Module):
    def __init__(self, n_inputs, n_states, n_outputs, n_hidden=64,
                 Ts=4.0, tau_min=4.0, tau_max=None):
        super().__init__()
        self.Ts = Ts
        self.log_k   = nn.Parameter(torch.zeros(4))
        self.cell    = LTCCell(n_inputs + 2, n_hidden, Ts, tau_min, tau_max)
        self.ic_h    = nn.Sequential(nn.Linear(n_states, n_hidden), nn.Tanh())
        self.ic_z    = nn.Linear(n_states, 2)
        self.out     = nn.Linear(n_hidden, n_outputs)

    def get_initial_hidden(self, y0):
        y0f = y0.squeeze(1) if y0.dim() == 3 else y0
        return self.ic_h(y0f), self.ic_z(y0f)

    def forward_from_hidden(self, u, hidden):
        h, z = hidden
        k = F.softplus(self.log_k)
        outs = []
        for t in range(u.shape[1]):
            ut = u[:, t]
            z1, z2 = z[:, :1], z[:, 1:]
            dz1 = -k[0]*torch.sqrt(torch.clamp(z1,min=0)) + k[3]*ut
            dz2 =  k[1]*torch.sqrt(torch.clamp(z1,min=0)) - k[2]*torch.sqrt(torch.clamp(z2,min=0))
            z = (z + self.Ts * torch.cat([dz1, dz2], -1)).detach()
            h = self.cell(torch.cat([ut, z], -1), h)
            outs.append(self.out(h))
        return torch.stack(outs, 1), (h, z)

    def forward(self, u, y0):
        return self.forward_from_hidden(u, self.get_initial_hidden(y0))


# ──────────────────────────────────────────────
# 7. Cascaded LTC (fast cell → slow cell)
# ──────────────────────────────────────────────

class CascadedLTCModel(nn.Module):
    def __init__(self, n_inputs, n_states, n_outputs, n_hidden=64,
                 Ts=4.0, tau_min=4.0, tau_max=None):
        super().__init__()
        h1, h2 = n_hidden // 2, n_hidden - n_hidden // 2
        self.h1, self.h2 = h1, h2
        self.cell1 = LTCCell(n_inputs, h1, Ts, tau_min, tau_max)
        self.cell2 = LTCCell(n_inputs + h1, h2, Ts, tau_min * 4, tau_max)
        self.ic1 = nn.Sequential(nn.Linear(n_states, h1), nn.Tanh())
        self.ic2 = nn.Sequential(nn.Linear(n_states, h2), nn.Tanh())
        self.out = nn.Linear(h2, n_outputs)

    def get_initial_hidden(self, y0):
        y0f = y0.squeeze(1) if y0.dim() == 3 else y0
        return self.ic1(y0f), self.ic2(y0f)

    def forward_from_hidden(self, u, hidden):
        h1, h2 = hidden
        outs = []
        for t in range(u.shape[1]):
            ut = u[:, t]
            h1 = self.cell1(ut, h1)
            h2 = self.cell2(torch.cat([ut, h1], -1), h2)
            outs.append(self.out(h2))
        return torch.stack(outs, 1), (h1, h2)

    def forward(self, u, y0):
        return self.forward_from_hidden(u, self.get_initial_hidden(y0))


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def build_model_from_config(config_pars, n_inputs, n_states, n_outputs):
    mt = str(config_pars.get("type", "RNN")).upper()
    hs = list(config_pars.get("hidden_sizes", [4]))
    act = config_pars.get("activation", "Tanh")
    dp  = config_pars.get("dropout_prob", 0.0)

    if mt in {"RNN", "GRU", "LSTM"}:
        return CascadedTankModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=hs, num_layers=config_pars.get("num_layers", 1),
            recurrent=mt, activation=act, dropout_prob=dp,
            direct_feedthrough=config_pars.get("direct_feedthrough", False),
        )
    if mt == "PHYSICAL_RK4":
        return PhysicalRK4Model(n_inputs, n_states, n_outputs,
                                Ts=config_pars.get("Ts", 4.0),
                                ic_hidden=config_pars.get("ic_hidden", 32))
    if mt in {"LTC", "CFC"}:
        return LTCModel(n_inputs, n_states, n_outputs,
                        n_hidden=config_pars.get("n_hidden_states", 64),
                        Ts=config_pars.get("Ts", 4.0),
                        tau_min=config_pars.get("tau_min", 4.0),
                        tau_max=config_pars.get("tau_max", None),
                        dropout_prob=dp)
    if mt == "DIAGONAL_SSM":
        return DiagonalSSMModel(n_inputs, n_states, n_outputs,
                                n_hidden=config_pars.get("n_hidden_states", 128))
    if mt == "CAUSAL_CNN":
        return CausalDilatedCNNModel(n_inputs, n_states, n_outputs,
                                     n_channels=config_pars.get("n_channels", 32),
                                     n_layers=config_pars.get("n_layers", 7))
    if mt == "HYBRID_LTC":
        return HybridLTCModel(n_inputs, n_states, n_outputs,
                              n_hidden=config_pars.get("n_hidden_states", 64),
                              Ts=config_pars.get("Ts", 4.0),
                              tau_min=config_pars.get("tau_min", 4.0),
                              tau_max=config_pars.get("tau_max", None))
    if mt == "CASCADED_LTC":
        return CascadedLTCModel(n_inputs, n_states, n_outputs,
                                n_hidden=config_pars.get("n_hidden_states", 64),
                                Ts=config_pars.get("Ts", 4.0),
                                tau_min=config_pars.get("tau_min", 4.0),
                                tau_max=config_pars.get("tau_max", None))
    raise ValueError(f"Unknown model type: {mt}")
