# Search Journal — autoresearch/gen01

Branch: autoresearch/gen01
Started: 2026-06-28

---

## Iteration 0 — Baseline

**Changed:** nothing (baseline run as-is)
**Hypothesis:** establish reference performance
**Result:** val_RMSE=0.6812; model is tiny RNN (n_hidden=4, hidden=[4]) with direct feedthrough. Very weak performance, large gap between train and val confirms the model lacks capacity to capture the tank nonlinear dynamics.
**Decision:** keep
**Next:** The baseline is very weak. First priority is a major architecture change. Try PHYSICAL_RK4 — it encodes the Torricelli ODE structure directly with only ~10 parameters. Expect better inductive bias for this physical system.

---

## Iteration 1 — PHYSICAL_RK4 grey-box

**Changed:** type=PHYSICAL_RK4, n_hidden_states=32, hidden_sizes=[32,32]
**Hypothesis:** Torricelli's law ODE with RK4 integration encodes true physics and should generalize better than a black-box RNN
**Result:** val_RMSE=1.0241; worse than baseline. The model overfits fold_2 — overflow effects and normalization mismatch prevent the physical ODE from generalizing across folds.
**Decision:** discard
**Next:** Reset to baseline. Pure physical model underperforms because the normalized-space dynamics do not match Torricelli's law directly. Try a large LSTM (n_hidden=64) with direct feedthrough — larger capacity black-box should capture the dynamics without hard physical assumptions.

---

## Iteration 2 — LSTM n_hidden=64 2-layer

**Changed:** type=LSTM, n_hidden_states=64, hidden_sizes=[64,64], Tanh, dropout=0.1, direct_feedthrough=True
**Hypothesis:** A 16× larger LSTM with gating should capture longer-range dynamics and improve generalization over the tiny RNN baseline
**Result:** val_RMSE=0.4969; major improvement over baseline (0.6812→0.497). The gating mechanism helps stability. Some overfitting remains (train < val) but generalization is much better.
**Decision:** keep
**Next:** Good improvement. Try HYBRID physics+LSTM residual — combine Torricelli backbone with learned residual to get both physical inductive bias and data-driven flexibility.

---

## Iteration 3 — HYBRID physics+LSTM residual

**Changed:** type=HYBRID, n_hidden_states=32, hidden_sizes=[32,32], physical backbone + LSTM residual
**Hypothesis:** Combining the Torricelli physical model as backbone with a learned LSTM residual should give best of both worlds
**Result:** val_RMSE=1.5814; much worse. The physical backbone overfits fold_2 — the physical model produces poor state estimates that confuse the LSTM residual, especially early in training when physical parameters are random.
**Decision:** discard
**Next:** Reset to LSTM-64. Hybrid approach fails because the normalized-space ODE is unreliable. Try LTC (Liquid Time-Constant / Closed-Form CTC) — it has unconditional stability via exp(-dt/tau) state updates and physically meaningful time constants that can match the tank dynamics (4–400 s range).

---

## Iteration 4 — LTC CfC n_hidden=64 dt=4s

**Changed:** type=LTC, n_hidden_states=64, hidden_sizes=[64,64], Tanh, dt=4.0, direct_feedthrough=True
**Hypothesis:** LTC's exponential-decay state updates A=exp(-dt/tau) provide unconditional stability and match the physical time scales of the tank system better than LSTM gating
**Result:** val_RMSE=0.2486; major breakthrough — from 0.497 (LSTM) to 0.249. Training is very stable and converges smoothly. The time-constant parameterization is well-suited to this system.
**Decision:** keep (new best)
**Next:** LTC is significantly better. Explore scaling: try n_hidden=128 to see if more capacity helps without overfitting.

---

## Iteration 5 — LTC n_hidden=128

**Changed:** n_hidden_states=128, hidden_sizes=[128,128]; otherwise same as iteration 4
**Hypothesis:** More capacity in LTC might capture more complex dynamics
**Result:** val_RMSE=0.3553; worse than LTC-64 (0.249). More capacity leads to overfitting — fold_2 performance degrades significantly.
**Decision:** discard
**Next:** Reset to LTC-64. n_hidden=64 is the capacity sweet spot. Try DynamicLTC with input-dependent gating A(h,u) — if the effective time constant varies with the input signal, this could model the nonlinear flow more accurately.

---

## Iteration 6 — DynamicLTC input-dependent gating

**Changed:** type=DYNAMIC_LTC, n_hidden_states=64, hidden_sizes=[64,64]; A matrix computed as function of (h,u)
**Hypothesis:** Input-dependent gating allows the effective time constant to adapt to the current operating point, better modeling the nonlinear sqrt flow
**Result:** val_RMSE=0.2917; worse than standard LTC-64 (0.249). More flexible gating leads to overfitting — the additional parameters are not justified by the dataset size.
**Decision:** discard
**Next:** Reset to LTC-64. The fixed tau parameterization is sufficient. Try freezing tau as a regularizer — if learned tau values are near-random, fixing them might act as an inductive bias toward a linear filter bank.

---

## Iteration 7 — LTC-64 frozen tau

**Changed:** type=LTC, n_hidden_states=64, tau frozen (not trained), otherwise same as iteration 4
**Hypothesis:** Frozen tau forces the model to use fixed exponential time constants as a linear filter bank; learned output mapping may generalize better
**Result:** val_RMSE=0.2517; essentially the same as trainable LTC-64 (0.249). Tau learning has negligible impact — the model finds similar tau values regardless.
**Decision:** discard
**Next:** Reset to LTC-64. Tau is not the source of overfitting. Try NeuralSSM2 — a 2-state neural state-space model with Euler residual dynamics, explicitly modeling z1 and z2 as physical states with a neural f function.

---

## Iteration 8 — NeuralSSM2 2-state Euler dynamics

**Changed:** type=NEURAL_SSM2, hidden_sizes=[16,16], IC encoder 10→2, f_dyn MLP [16,16], Euler integration, lr=3e-3
**Hypothesis:** A 2-state SSM explicitly modeling the two tank levels with Euler residual h_{t+1}=h_t+f(h_t,u_t) has better physical inductive bias
**Result:** val_RMSE=0.4319; much worse than LTC-64 (0.249). Euler integration is numerically unstable — no gating to prevent state growth. Train RMSE=0.300 but val=0.432 shows overfitting.
**Decision:** discard
**Next:** Reset to LTC-64. Euler residual dynamics are unstable without gating. Try reducing LTC f_net/ic_net capacity to hidden=[32] — test if the MLP sub-networks are the overfitting bottleneck.

---

## Iteration 9 — LTC-64 smaller f_net/ic_net hidden=[32]

**Changed:** type=LTC, n_hidden_states=64, hidden_sizes=[32] (smaller MLP sub-networks), lr=1e-3
**Hypothesis:** Smaller f_net and ic_net reduce parameter count and overfitting without hurting representational capacity
**Result:** val_RMSE=0.3962; worse than full LTC-64 (0.249). The smaller MLP underfits — hidden=[64] is necessary to capture the sqrt nonlinearity.
**Decision:** discard
**Next:** Reset to LTC-64. MLP capacity at [64,64] is needed. Try ESN (Echo State Network) — a fundamentally different approach with a fixed random reservoir and only a linear readout trained, as a strong regularization baseline.

---

## Iteration 10 — ESN fixed random reservoir n=500

**Changed:** type=ESN, n_hidden_states=500, spectral_radius=0.95, only linear readout trained, lr=1e-2
**Hypothesis:** A large fixed reservoir with only a linear readout is maximally regularized — if it works, it suggests the dynamics are nearly linear
**Result:** val_RMSE=0.8838; much worse than LTC-64. The linear readout is insufficient for the nonlinear sqrt tank dynamics. The reservoir has rich dynamics but cannot be extracted by a linear mapping.
**Decision:** discard
**Next:** Reset to LTC-64. System is genuinely nonlinear. Try TCN (Temporal Convolutional Network) — dilated causal convolutions can capture long-range dependencies with a large receptive field.

---

## Iteration 11 — TCN 8-layer dilated causal conv

**Changed:** type=TCN, n_hidden_states=64, 8 layers, kernel=3, dilation=1..128, receptive field=511, lr=1e-3
**Hypothesis:** Dilated causal convolutions with large receptive field capture long-range temporal dependencies more efficiently than recurrent models
**Result:** val_RMSE=0.6617; severe overfitting (train=0.013, val=0.784 at best epoch). The TCN memorizes the training sequence perfectly but lacks temporal stability — without fixed time constants it cannot generalize to the other fold.
**Decision:** discard
**Next:** Reset to LTC-64. TCN overfits due to lack of recurrent state stability. Try StackedLTC — a 2-stage cascade where stage1 has fast time constants [4,20]s and stage2 has slow constants [20,100]s, matching the two physical tank time scales.

---

## Iteration 12 — StackedLTC 2-stage cascade [32+32]

**Changed:** type=STACKED_LTC, 2 stages (32+32 hidden), stage1 tau=[4,20]s, stage2 tau=[20,100]s, hidden_sizes=[64,64], dt=4.0
**Hypothesis:** Two LTC stages with different time constant ranges explicitly model the two-tank cascade dynamics (fast upper tank, slow lower tank)
**Result:** val_RMSE=0.3121; balanced folds (fold_1=0.303, fold_2=0.320) but worse than single LTC-64 (0.249). The cascade constraint is too rigid — direct input→output path is needed.
**Decision:** discard
**Next:** Reset to LTC-64. The stacked structure does not improve over the single LTC. Try LTC-64 with dropout=0.2 — stochastic regularization in f_net and ic_net to reduce overfitting on fold_2.

---

## Iteration 13 — LTC-64 dropout=0.2

**Changed:** type=LTC, n_hidden_states=64, dropout_prob=0.2 in f_net and ic_net
**Hypothesis:** Dropout regularization in the sub-networks reduces overfitting on fold_2 without losing capacity
**Result:** val_RMSE=0.3022; balanced folds (fold_1=0.284, fold_2=0.320) but worse than no-dropout LTC-64 (0.249). Dropout hurts fold_1 significantly — the model is not overfitting through the MLP weights.
**Decision:** discard
**Next:** Reset to LTC-64. Dropout is counterproductive. Try hidden state noise std=0.05 — state-level noise injection during training as an alternative regularizer that acts on the recurrent dynamics directly.

---

## Iteration 14 — LTC-64 hidden state noise std=0.05

**Changed:** type=LTC, n_hidden_states=64, hidden_noise_std=0.05 (Gaussian noise added to hidden state during training)
**Hypothesis:** State-level noise acts as a regularizer on the recurrent dynamics, improving robustness to fold distribution shift
**Result:** val_RMSE=0.3152; worse than clean LTC-64 (0.249). State noise hurts fold_1 (0.290 vs 0.176) without helping fold_2. The noise disrupts the stable time-constant dynamics that make LTC effective.
**Decision:** discard
**Final best:** LTC CfC n_hidden=64 dt=4s (iteration 4), val_RMSE=0.249
