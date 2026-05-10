# Search Journal — gen21

Branch: autoresearch/gen21
Started: 2026-05-02

---

## Iteration 0 — Baseline

**Changed:** nothing (baseline run as-is)
**Hypothesis:** establish reference performance for gen21 branch
**Result:** val_RMSE=0.6812, test_RMSE=1.4454; model is tiny RNN (n_hidden=4, hidden=[4]) with direct feedthrough. Training is unstable (oscillates after epoch 375 for fold_2), early stopping fires around epoch 337 median.
**Decision:** keep
**Next:** The baseline is very weak. The model is tiny and training is unstable. The first priority is a major architecture change. The cascaded tank has known physics (Torricelli sqrt flow). Try the PhysicalTankModelRK4 (already in model.py) — it encodes the ODE structure directly with only ~10 parameters. Expect much better inductive bias for this system.

## Iteration 1 — PHYSICAL_RK4 grey-box

**Changed:** type=PHYSICAL_RK4, lr=5e-3, max_epochs=3000, early_stopping_patience=20, hidden_sizes=[32,32], Tanh
**Hypothesis:** Torricelli's law ODE with RK4 integration encodes true physics, should generalize better than black-box RNN
**Result:** val_RMSE=1.0243, test_RMSE=1.6844; worse than baseline. Train RMSE=0.505 but val is 1.024 — the model overfits to training fold or the normalized-space ODE doesn't generalize to the other fold.
**Decision:** discard
**Next:** Reset to baseline. The pure physical model underperforms because: (a) the normalized-space dynamics don't match Torricelli's law directly, (b) overflow effects are not well captured. Try the HYBRID model (PhysicalTankModel + LSTM residual) which combines physics with learned corrections, or try a larger LSTM (n_hidden=64) with better initialization. The key issue is the model is too small. Try LSTM with n_hidden=64, deeper init network, and multi-step prediction loss (truncated BPTT) which has been shown to help for dynamical systems.

## Iteration 2 — Large LSTM + TBPTT

**Changed:** LSTM n_hidden=128, num_layers=2, hidden_sizes=[128,128], Tanh, dropout=0.1, lr=3e-4, patience=15, TBPTT chunk_size=50, weight_decay=1e-5
**Hypothesis:** The baseline model was vastly too small (4 hidden). A 32x larger LSTM with truncated BPTT should capture the longer-range dynamics and improve generalization
**Result:** val_RMSE=0.4298, test_RMSE=1.3494; massive improvement over baseline (0.6812). Best improvement yet. The model overfits training (train RMSE=0.146 vs val=0.430) but val is still much better.
**Decision:** keep (new best)
**Next:** Validation RMSE is 0.430. Training RMSE is 0.146, showing significant overfitting. Options: (a) increase regularization (higher dropout, weight decay), (b) reduce model size slightly (n_hidden=64), (c) try noise injection during training (Gaussian noise on inputs/targets to act as regularizer), (d) try a different model family. Let me try a Neural State-Space model (NSSM) — a model that explicitly maintains 2D physical state [z1, z2] but with neural f and g functions. This has better inductive bias than a generic LSTM.

## Iteration 3 — Neural SSM with 8D residual dynamics

**Changed:** type=NSSM, n_hidden_states=8, hidden_sizes=[64,64], Tanh, lr=1e-3, TBPTT chunk=50, weight_decay=1e-4, patience=20
**Hypothesis:** A small-dimensional state-space model with Euler residual dynamics should have better inductive bias than LSTM and less overfitting
**Result:** val_RMSE=0.768, test_RMSE=2.457; much worse than LSTM-128. Training was very unstable (loss exploding/oscillating). The simple Euler residual h_{t+1}=h_t+f(h_t,u_t) is numerically unstable — no gating to prevent state growth.
**Decision:** discard
**Next:** Reset to LSTM-128 (089a233). The LSTM architecture works much better because of the gating mechanism providing stability. Focus on reducing overfitting of LSTM: (a) try LTC model which has unconditional stability via exp(-dt/tau) gates, or (b) add input noise regularization to LSTM-128, or (c) try LSTM-64 with stronger regularization. Try LTC (Closed-Form CTC) — it has physically meaningful time constants matching tank dynamics.

## Iteration 4 — LTC (Closed-Form CTC) n_hidden=64

**Changed:** type=LTC, n_hidden_states=64, hidden_sizes=[64,64], Tanh, dt=4.0, lr=1e-3, weight_decay=1e-4, patience=20, no TBPTT
**Hypothesis:** LTC's exponential-decay state updates (A=exp(-dt/tau)) provide unconditional stability and match the physical time scales of the tank system better than LSTM gating
**Result:** val_RMSE=0.2500, test_RMSE=0.2977; major improvement from 0.430 (LSTM) to 0.250. Training is very stable and converges smoothly. The time-constant parameterization appears well-suited to this system.
**Decision:** keep (new best)
**Next:** The LTC is significantly better. Now refine: (a) try larger LTC (n_hidden=128) to see if more capacity helps without overfitting, (b) try direct_feedthrough=False since it's already True and may not be needed, (c) try lower lr or longer training to squeeze more improvement. Start with n_hidden=128 to test capacity scaling.

## Iteration 5 — LTC n_hidden=128

**Changed:** n_hidden_states=128, hidden_sizes=[128,128]; otherwise same as iteration 4
**Hypothesis:** More capacity in LTC might capture more complex dynamics
**Result:** val_RMSE=0.355, test_RMSE=0.421; worse than LTC-64 (0.250). More capacity leads to overfitting.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). The sweet spot is at n_hidden=64. Try n_hidden=32 with stronger regularization to reduce overfitting. Also try modifying the LTC to use a physically-informed initialization for tau — set tau range to match tank dynamics better. The tank has time constants roughly between 4s (1 step) and 400s (100 steps), so the range [4, 400] makes more sense than [4, 100].

## Iteration 6 — LTC-64 wider tau + dropout + stronger weight decay

**Changed:** tau range [4,500s], dropout=0.1, weight_decay=2e-4, lr=5e-4, patience=25
**Hypothesis:** Wider tau range covers slower tank dynamics; dropout+wd reduces overfitting
**Result:** val_RMSE=0.287; slightly worse than iteration 4 (0.250). Dropout hurts more than it helps — fold_1 gets worse with dropout. The regularization is too strong relative to the small model size.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). Try two things: (a) Hybrid Physical+LTC — physics provides backbone, LTC corrects residuals; (b) LTC-64 with just wider tau and no dropout. The key insight from iteration 4 logs: fold_2's val RMSE increases after epoch 75 (from ~0.302 to 0.405), suggesting the model overfits fold_2 during later training. The best approach might be a smaller model with inherently better generalization.

## Iteration 7 — LTC-64 with input noise std=0.05

**Changed:** Added input_noise_std=0.05 to config_pars; noise added to u during training
**Hypothesis:** Input noise acts as a regularizer to prevent memorizing specific input patterns
**Result:** val_RMSE=0.251, essentially the same as iteration 4 (0.250). Input noise doesn't help or hurt.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). Try Koopman operator model — lifts dynamics to linear space where generalization may be easier. The Koopman approach has well-founded theoretical basis for nonlinear dynamical systems.

## Iteration 8 — Diagonal SSM with linear dynamics

**Changed:** type=DSSM, strictly linear dynamics x[t+1]=A*x+B*u, A=diag(exp(-dt/tau)), tau in [4,500s], linear readout
**Hypothesis:** Linear dynamics in lifted space might generalize better than nonlinear LTC
**Result:** val_RMSE=0.364; much worse than LTC-64 (0.250). The purely linear dynamics cannot capture the sqrt nonlinearity of the Torricelli outflow. The f_net nonlinearity in LTC is essential.
**Decision:** discard
**Next:** Reset to LTC-64. The nonlinear f_net in LTC is crucial. Try two directions: (a) LTC-64 with SiLU activation + AdamW optimizer; (b) a GRU with physics-motivated bias initialization. The GRU with LTC-like initialization might combine adaptive gating with temporal inductive bias.

## Iteration 9 — GRU-64 with LTC-like gate initialization

**Changed:** type=GRU, n_hidden=64, TBPTT chunk=50, ltc_gate_init=True
**Hypothesis:** GRU with physics-motivated initialization should benefit from adaptive gating while starting from physically meaningful time constants
**Result:** val_RMSE=0.388; much worse than LTC-64 (0.250). Adaptive gating leads to overfitting.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). Try a cascaded two-timescale LTC that mirrors the physical cascade: fast LTC for tank-1 (driven by u), slow LTC for tank-2 (driven by fast LTC). This enforces physics without hard-coding parameters.


## Iteration 10 — CascadedLTC two-timescale

**Changed:** type=CASCADED_LTC, n_hidden=64 (32 fast tau:[4,30s], 32 slow tau:[30,500s]), f_slow driven by h_fast not u
**Hypothesis:** Explicit cascade structure matching tank1/tank2 dynamics should improve generalization
**Result:** val_RMSE=0.286; worse than LTC-64 (0.250). The cascade constraint is too rigid — fold_1 val plateaus at 0.280 while LTC-64 achieves 0.176. Direct u→output path is needed.
**Decision:** discard
**Next:** Reset to LTC-64. Try HybridLTC: physical Euler model provides z1,z2 state estimates as additional features to the LTC (input dim 1→3). The LTC can leverage physics-based state features while correcting them. Different from previous hybrid attempts where physics was a backbone with LSTM correction.


## Iteration 11 — HybridLTC (physics z1/z2 as LTC augmented input)

**Changed:** New HybridLTC class: runs Euler Torricelli physics (k1-k4 learnable) in parallel with LTC-64; at each step concatenates [u_t, z1_t, z2_t] as LTC input (dim 1→3). After NaN explosion with k=1 init, fixed with log_k=-2.0 init and detached physics states in LTC recurrence (direct readout still sees z1,z2).
**Hypothesis:** Physical state estimates z1,z2 provide inductive bias as auxiliary features, letting LTC correct residuals from a physically-informed baseline
**Result:** val_RMSE=0.808, test_RMSE=1.328; much worse than LTC-64 (0.250). The detached physics features act as noise for the LTC — the physics model produces poor z1/z2 estimates early in training (random init) that confuse the LTC, and even after convergence the physics estimate quality is limited.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). Try a completely different architectural approach: WaveNet-like causal dilated CNN. This processes the full input sequence u in one pass using causal (past-only) dilated convolutions. Receptive field = 63 steps = 252s, covering all relevant time constants. No recurrence = no BPTT gradient issues. y0 injected as a constant auxiliary channel. This is a major departure from all recurrent approaches tried so far.

## Iteration 12 — CausalDilatedCNN (WaveNet-style, 7 layers, RF=127 steps)

**Changed:** New CausalDilatedCNN: 7 layers of causal dilated 1D convolutions (dilation 1,2,4,...,64), receptive field 127 steps (508s at Ts=4s). y0 injected as constant auxiliary channel. Residual connections. No recurrence. Direct feedthrough. n_channels=32, ~15.7K params.
**Hypothesis:** Causal CNN avoids BPTT gradient issues entirely; with 127-step receptive field it captures all relevant dynamics; no trainable time constants avoids the implicit regularization trade-off
**Result:** val_RMSE=0.544, test_RMSE=1.102; much worse than LTC-64 (0.250). Extreme overfitting: train_RMSE=0.024 vs val=0.708. Best epoch was 25, then validation diverged. The CNN lacks temporal stability — without fixed time constants, it memorizes the training sequence perfectly but doesn't generalize.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). LTC's fixed time constants are an essential regularizer for this problem. Try LTC with n_hidden=32 — smaller model may be more regularized than n_hidden=64 (which was better than n_hidden=128). This refinement was planned after iteration 5 but never executed.

## Iteration 13 — LTC n_hidden=32

**Changed:** n_hidden_states=32 (from 64), hidden_sizes=[32,32]
**Hypothesis:** Smaller LTC might generalize better — n_hidden=64 may be slightly over-parameterized
**Result:** val_RMSE=0.372, test_RMSE=0.666; worse than LTC-64 (0.250). fold_2 best at epoch 100 (val=0.353), then diverged. n_hidden=32 lacks capacity to capture sqrt nonlinearity. Confirms 64 is the sweet spot.
**Decision:** discard
**Next:** Reset to LTC-64. The architecture sweet spot is confirmed at n_hidden=64. Try training dynamics improvements: AdamW optimizer + CosineAnnealingLR instead of Adam+ReduceLROnPlateau. Cosine annealing provides smoother LR decay and AdamW correctly decouples weight decay from gradient moments.

## Iteration 14 — LTC-64 AdamW + CosineAnnealingLR + patience=30

**Changed:** Adam→AdamW, ReduceLROnPlateau→CosineAnnealingLR(T_max=3000, eta_min=1e-5), patience=20→30; scheduler now steps every epoch
**Hypothesis:** AdamW decouples weight decay correctly; cosine annealing provides smoother LR decay allowing longer useful training
**Result:** val_RMSE=0.254, test_RMSE=0.301; slightly worse than LTC-64 baseline (0.250). fold_1 improved (0.182 vs 0.176) but fold_2 worse (0.327 vs 0.302). Cosine annealing keeps LR high early, causing fold_2 to overfit before LR drops. ReduceLROnPlateau's aggressive LR reduction is actually beneficial for fold_2.
**Decision:** discard
**Next:** Reset to LTC-64 (eebe0df). Try LTC-64 with SiLU activation (replacing Tanh). SiLU (x*sigmoid(x)) is unbounded above and has smoother gradients — may better approximate the sqrt Torricelli nonlinearity.


