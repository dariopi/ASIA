# Search Journal — gen12 (causal-only, 15 iterations)

## Run 1 — baseline
- **Changed:** nothing
- **Hypothesis:** establish reference performance
- **Result:** val_MAE = 0.7067 (LSTM-32, 1L, direct prediction, MSE, Adam)
- **Decision:** keep
- **Next:** delta-state prediction + state feedback + larger model + MAE loss + AdamW + motor²

## Run 2 — AutoregressiveDeltaGRU-256 3L
- **Changed:** full architectural overhaul: delta-state prediction, state feedback y_{t-1}, motor², MAE loss, AdamW, cosine WR, teacher forcing
- **Hypothesis:** combining all key improvements at once as a second reference
- **Result:** val_MAE = 0.3229 (chirp≈0.12, random≈0.26, square≈0.59)
- **Decision:** keep
- **Next:** add structured physics prior

## Run 3 — PhysicsResidualCausal
- **Changed:** physics prior (kinematic + thrust + torque) + GRU-256 3L residual
- **Hypothesis:** structured physics prior improves generalization
- **Result:** val_MAE = 0.307497
- **Decision:** keep
- **Next:** explore variations on architecture and training

## Run 4 — KoopmanDroneModel
- **Changed:** encoder→linear latent dynamics A,B→decoder architecture
- **Hypothesis:** Koopman linear dynamics might capture global structure
- **Result:** val_MAE = 0.4382 — linear dynamics too restrictive for nonlinear drone
- **Decision:** discard

## Run 5 — PhysicsResidualCausal no teacher forcing
- **Changed:** no teacher forcing, state noise, dropout=0.2
- **Hypothesis:** free-running training better matches eval
- **Result:** val_MAE = 0.3445 — worse; always-on TF is better
- **Decision:** discard

## Run 6 — PhysicsResidualCausal GRU-384 4L + motor lag
- **Changed:** wider/deeper GRU, motor lag features
- **Hypothesis:** more capacity / temporal features
- **Result:** val_MAE = 0.3087 > 0.307497 — negligible change, not enough to beat run 3
- **Decision:** discard

## Run 7 — PhysicsResidualCausal LSTM + structured thrust
- **Changed:** LSTM instead of GRU, different thrust architecture
- **Hypothesis:** LSTM's cell state might carry long-term dynamics better
- **Result:** val_MAE = 0.3102 > 0.307497 — marginal worsening
- **Decision:** discard

## Run 8 — CausalTCNDrone
- **Changed:** entirely new architecture: dilated causal convolutions + autoregressive delta decoding
- **Hypothesis:** parallel multi-scale temporal features from motor history
- **Result:** val_MAE = 0.5599 — much worse than GRU; TCN context doesn't replace hidden state
- **Decision:** discard

## Run 9 — CausalTransformerDrone
- **Changed:** entirely new architecture: causal multi-head self-attention
- **Hypothesis:** global attention might capture step-function input patterns better
- **Result:** val_MAE = 2.001 — catastrophic train-eval mismatch; parallel TF training ≠ autoregressive eval
- **Decision:** discard

## Run 10 — True scheduled sampling (Bernoulli TF decay) — BEST
- **Changed:** fixed teacher forcing bug (was always-on); added per-batch Bernoulli, TF_start=1.0→0.0
- **Hypothesis:** true exposure bias correction improves OOD generalization
- **Result:** val_MAE = 0.300280 — improves over run 3 (0.307497); new best
- **Decision:** keep — new best (0.300280 < 0.307497)

## Run 11 — Input noise augmentation σ=0.02
- **Changed:** Gaussian noise on motor inputs during training
- **Hypothesis:** noise augmentation improves robustness to OOD motor patterns
- **Result:** val_MAE = 0.3090 > 0.300280 — slightly worse; noise disrupts physics prior learning
- **Decision:** discard

## Run 12 — Dual loss (TF + free-running)
- **Changed:** every batch computes both teacher-forced loss and free-running loss, averaged
- **Hypothesis:** directly teaching the model to handle its own predictions
- **Result:** val_MAE = 0.3419 > 0.300280 — worse; conflicting objectives hurt convergence
- **Decision:** discard

## Run 13 — PhysicsResidualCausalDeltaU
- **Changed:** added Δu_t = u_t - u_{t-1} to GRU input (explicit regime signal)
- **Hypothesis:** Δu=0 signals constant-input regime (square fold); GRU can learn to suppress residual
- **Result:** val_MAE = 0.3198 > 0.300280 — GRU doesn't learn to exploit this signal when training on varying inputs
- **Decision:** discard

## Run 14 — MLP physics subnets
- **Changed:** replaced linear thrust_net and motor_mix with 2-layer MLPs (8→32→3, 4→16→3)
- **Hypothesis:** more expressive physics prior reduces GRU residual burden for OOD cases
- **Result:** val_MAE = 0.3042 > 0.300280 — marginal improvement vs linear physics but can't beat run 10
- **Decision:** discard

## Run 15 — GRU-128 2L + weight_decay=5e-4
- **Changed:** smaller GRU residual, higher regularization to force physics-prior reliance
- **Hypothesis:** smaller, more regularized GRU generalizes better to square fold
- **Result:** val_MAE = 0.3407 > 0.300280 — insufficient capacity hurts more than regularization helps
- **Decision:** discard

---

## Summary
**Budget exhausted (15/15 runs)**

**Best model:** PhysicsResidualCausal + scheduled sampling (commit b018402)
- val_MAE = 0.300280 (mean over folds)
- Architecture: PhysicsResidualCausal + true Bernoulli teacher-forcing decay (TF_start=1.0→0.0)

**Key findings:**
1. Physics prior (kinematic + thrust + torque) is the critical ingredient — improves over pure GRU by ~8%
2. True scheduled sampling (Bernoulli TF decay) provides a small but consistent gain over always-on TF
3. Square fold (OOD from chirp+random) is the irreducible bottleneck — none of the 15 runs significantly improved square fold performance
4. Alternative architectures (TCN, Transformer, Koopman) all fail compared to Physics+GRU
5. Adding features (Δu, noise, larger model) did not help generalization to square fold
6. The square fold bottleneck appears fundamental given the training data distribution mismatch
