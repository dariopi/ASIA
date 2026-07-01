# Search Journal — autoresearch/gen09

## Summary
- 15 runs, 4 kept, best: GRU n_hidden=256 num_layers=2 TBPTT AdamW CosineWarmRestarts (0.243113)
- Key finding: GRU with TBPTT + AdamW + CosineWarmRestarts wins. Fold-2 is the persistent bottleneck.

---

## [15/15] fdb06a9 — NeuralODE n_ode_states=16 ode_hidden=128 RK4 SiLU
- fold1=NaN, fold2=NaN, mean=NaN — DISCARD
- NaN from epoch 0: RK4 sub-steps with Ts=4.0 diverge immediately; f_net (MLP) unconstrained dx/dt blows up before grad_clip can help; need sub-stepping or output normalization

## [14/15] c5a77f4 — KoopmanOperator n_koopman=256 encoder_hidden=128
- fold1=0.738724 (pre-NaN), fold2=1.137532 (pre-NaN), mean=0.938128 — DISCARD
- NaN explosion: unconstrained K matrix (256×256 linear) has eigenvalues > 1, diverges over T~1000 steps; need spectral norm or leaky damping

## [13/15] a78a223 — LSTM n_hidden=256 num_layers=3 AdamW CosineWarmRestarts TBPTT chunk=50
- fold1=0.220762, fold2=0.378821, mean=0.299791 — DISCARD (vs GRU-256 0.243)
- Third LSTM layer hurts; more depth → worse generalization on fold_2; GRU simpler gating wins

## [12/15] 6d7dc30 — EchoStateNetwork reservoir_size=500 spectral_radius=0.95 leak_rate=0.3
- fold1=0.205355, fold2=0.412507, mean=0.308931 — DISCARD (vs GRU-256 0.243)
- fold_1 good (0.205) but fold_2 bottleneck (0.413); linear readout can't capture sqrt nonlinearity; reservoir features not aligned with tank dynamics

## [11/15] 600c6ce — CausalTransformer d_model=64 n_heads=4 n_layers=3
- fold1=0.721312, fold2=0.748365, mean=0.734838 — DISCARD
- Pure attention overfits badly: memorizes single training sequence but fails to generalize; train RMSE low but val >> best

## [10/15] 5ab211d — GRU n_hidden=256 num_layers=2 AdamW CosineWarmRestarts TBPTT chunk=50 *** BEST ***
- fold1=0.184068, fold2=0.302157, mean=0.243113 — KEEP
- Scaling GRU from 128→256 hidden + AdamW optimizer + cosine LR schedule beats previous best; fold_2 remains bottleneck

## [9/15] a03c508 — CascadedLTC n_hidden=64 Ts=4.0 TBPTT chunk=50
- mean=0.416839 — DISCARD (vs GRU-128 0.266)
- Two-timescale LTC (fast 32 + slow 32 units) not competitive with GRU

## [8/15] 5765655 — GRU n_hidden=128 num_layers=2 TBPTT chunk=50
- fold1=0.182493, fold2=0.350498, mean=0.266495 — KEEP
- GRU simpler gating than LSTM → better regularization; fold_1 excellent (0.182), fold_2 still bottleneck

## [7/15] bab09b0 — HybridLTC n_hidden=64 Ts=4.0 TBPTT chunk=50
- mean=0.995545, best@epoch37 — DISCARD
- Detached physics Euler z1/z2 context + TBPTT kills gradient signal to physics params log_k; model fails to learn useful dynamics

## [6/15] 11d6127 — CausalDilatedCNN n_channels=32 n_layers=7 RF=128
- mean=0.982152, best@epoch1 — DISCARD (no learning)
- CNN with gated activation + IC bias from y0; likely poor gradient signal for causal conv without normalization trick

## [5/15] 7de060b — DiagonalSSM n_hidden=128 TBPTT chunk=50
- fold1≈0.65, fold2≈0.66, mean=0.653713 — DISCARD (barely beats baseline)
- Linear SSM with tanh-diagonal A cannot capture Torricelli sqrt nonlinearity

## [4/15] d7cd03c — CFC n_hidden=64 Ts=4.0 tau_min=4.0 (full-seq backprop)
- fold1=0.340033, fold2=0.323761, mean=0.331897 — DISCARD (vs LSTM-128 0.309)
- Folds more balanced than before but still worse than LSTM; CfC needs more tuning

## [3/15] 8949c85 — LSTM n_hidden=128 num_layers=2 TBPTT chunk=50
- fold1=0.267471, fold2=0.350794, mean=0.309133 — KEEP (new best at the time)
- Deep LSTM + TBPTT; fold_2 remains bottleneck

## [2/15] 9ef3401 — PHYSICAL_RK4 (Torricelli ODE + RK4)
- fold1=1.132865, fold2=1.153791, mean=1.143328 — DISCARD
- sqrt structure breaks under normalization; physics inductive bias unhelpful in norm space

## [1/15] cedc2b1 — baseline RNN n_hidden=4
- fold1=0.726336, fold2=0.636045, mean=0.681190 — KEEP
- Tiny RNN, direct feedthrough, Adam lr=1e-3
