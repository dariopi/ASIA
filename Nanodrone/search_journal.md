# Search Journal — gen03

## Iteration 0 — Baseline

**Commit:** 1fd5e58

**What:** Baseline: LSTM-32, 1 layer, lr=2e-3, MSE loss, no mini-batch, no noise.

**Why:** Reference point for gen03.

**Result:** val_MAE=0.7067, test_MAE=0.2771. Same as gen02 baseline. LSTM-32 has too little capacity and MSE loss is a poor proxy for MAE.

**Status:** keep

**Next:** Jump immediately to full quadrotor physics model + LSTM residual. Use motor omega^2 thrust, rotation matrix, Euler kinematics, angular dynamics — all with learnable scale parameters. Add all training improvements from gen02 (mini-batch, MAE+multi-horizon loss, noise augmentation, AdamW+cosine). Hypothesis: explicit physics structure will generalize to square fold better than black-box LSTM.

---

## Iteration 1 — Quadrotor Physics Model + LSTM Residual

**Commit:** 85145d1

**What:** Full physical pipeline: T_i=k_T*omega_i^2, rotation matrix R(roll,pitch,yaw), Euler-angle kinematics, angular dynamics with inertia. All parameters learnable. LSTM residual for corrections. Plus all gen02 training improvements (mini-batch, MAE+multi-horizon, AdamW+cosine, noise).

**Why:** Explicit physics encodes the quadrotor flight equations, which should generalize better to unseen trajectory types (square fold).

**Result:** val_MAE=0.4470 (worse than baseline), test=0.1434. Square fold: 0.798. The problem is that angles are normalized and using cos/sin with a single learnable scale factor doesn't adequately represent the rotation matrix in normalized space. Large initial loss (26-32) and instability. The physics parametrization in normalized space didn't converge well.

**Status:** discard — reset to 1fd5e58

**Next:** Try KinematicsLSTMModel (gen02 proven approach: kinematic integration + residual velocity dynamics + full-state input) plus motor^2 as additional input feature. Gen02 best was 0.3406 with this approach; adding u^2 may help by providing direct thrust signal.

---

## Iteration 2 — Kinematic LSTM + Motor^2 Features

**Commit:** 76a0011

**What:** KinematicsLSTMModel (from gen02) with residual delta_v/delta_omega, full-state LSTM input, kinematic integration. New: motor^2 appended to motor inputs (8D total). All gen02 training: mini-batch, MAE+multi-horizon, AdamW+cosine, noise aug, dropout=0.1.

**Why:** Motor thrust is quadratic in motor speed; providing u^2 directly gives LSTM the thrust signal without needing to learn squaring. Hypothesis: better physical feature helps square fold generalization.

**Result:** val_MAE=0.3407 (chirp=0.146, random=0.255, square=0.621). Essentially tied with gen02 best (0.3406). Square fold improved slightly (0.621 vs 0.643) but random slightly worse (0.255 vs 0.240). Motor^2 feature provides marginal benefit.

**Status:** keep (new gen03 best)

**Next:** Try Koopman operator model with kinematic structure: linear dynamics in lifted state space for velocities (z_{t+1} = A@z_t + B@[u,u^2]) + kinematic integration for positions. Hypothesis: linear Koopman dynamics in lifted space may generalize better than LSTM to unseen trajectory types.

---

## Iteration 3 — Koopman Kinematic Model

**Commit:** 121251d

**What:** Linear Koopman dynamics z_{t+1}=A@z_t+B@[u,u^2] in lifted state (128D) + kinematic integration for positions. Linear velocity decoder C@z_t -> [v,omega].

**Why:** Linear dynamics in lifted space should generalize better and be more physically principled. Koopman theory guarantees any nonlinear system can be represented as linear in infinite-dimensional space; n_lifted=128 is finite approximation.

**Result:** val_MAE=0.4566 (chirp=0.194, random=0.375, square=0.801). Significantly worse than LSTM (0.3407). Linear dynamics insufficient to capture quadrotor nonlinearities at 128D lifting dimension. Square fold especially bad (0.801).

**Status:** discard — reset to 76a0011

**Next:** Causal Transformer model. No-feedback formulation: tokens=[y0_enc, u_1_enc,...,u_50_enc], causal mask, output at position t predicts full y_t. The Transformer can attend to full motor command history from y0 to reconstruct state. Completely different from LSTM — attention mechanism may capture dynamics patterns that generalize better to square.

---

## Iteration 4 — Causal Transformer (no state feedback)

**Commit:** a70bf76

**What:** Tokens=[y0_enc, u_1_enc,...,u_50_enc], causal mask, parallel processing. Output: delta_v, delta_omega per step. Kinematic integration. d_model=128, 3 layers, 4 heads.

**Why:** All motor commands known upfront; Transformer can attend over full motor history at each step without sequential state feedback. Avoids error accumulation from autoregressive state.

**Result:** val_MAE=0.4124 (chirp=0.195, random=0.271, square=0.771). Worse than LSTM. Slow to converge (median best=275). Without explicit state feedback, model struggles to track accumulated velocity changes. Model must implicitly reconstruct the current velocity from motor history, which is a harder learning problem.

**Status:** discard — reset to 76a0011

**Next:** Bidirectional encoder + state-feedback decoder. Since all 50 motor commands are known upfront, use a bidirectional LSTM to encode the full motor sequence (non-causal). Then at each step, combine bidirectional context with current state to predict dynamics. The bidirectional encoder gives the model "look-ahead" into future motor commands, which directly helps predict constant-velocity segments (square fold).

---


## Iteration 5 — Bidirectional Encoder + State-Feedback Decoder

**Commit:** b0cad59

**What:** BiEncoderKinematicModel: bidirectional LSTM encodes full motor sequence (look-ahead context), then per-step decoder combines bidirectional context with current state to predict delta_v/delta_omega. Kinematic integration as in iter2.

**Why:** All 50 motor commands known upfront; bidirectional encoding gives each step access to future motor commands. Should help constant-velocity segments (square fold) where the motor pattern is predictable from future context.

**Result:** val_MAE=0.3633 (chirp~0.150, random=0.296, square=0.641). Square marginally better (0.641 vs 0.621) but random significantly worse (0.296 vs 0.255). Net worse than LSTM best (0.363 vs 0.341). Bidirectional look-ahead hurt random fold generalization.

**Status:** discard — reset to 76a0011

**Next:** Scheduled sampling (teacher forcing decay) on top of proven KinematicsLSTMModel. Decay teacher_ratio from 0.3→0.0 over training epochs. At each rollout step t with probability teacher_ratio, use ground truth y_{t-1} instead of model's own prediction. Hypothesis: reduces train/test discrepancy for multi-step prediction, particularly in square fold.

---

## Iteration 6 — Scheduled Sampling (Teacher Forcing Decay)

**Commit:** a7b54a4

**What:** Scheduled sampling on KinematicsLSTMModel: teacher_ratio decays linearly from 0.3 to 0.0 over training epochs. At each rollout step t (t>0), with probability teacher_ratio, use ground truth y_{t-1} instead of model's own prediction. Config unchanged otherwise.

**Why:** Multi-step rollout computes training loss differently from inference: during training with full teacher forcing, model never sees its own errors; without it, errors accumulate. Linear decay bridges the gap. Hypothesis: reduces train/test discrepancy, helping square fold where constant-velocity error compounds.

**Result:** val_MAE=0.3096 (chirp=0.149, random=0.193, square=0.587). New best, beating 0.3407. All three folds improved: square 0.587 (from 0.621), random 0.193 (from 0.255), chirp 0.149 (from 0.146). Test ensemble MAE=0.090118.

**Status:** keep (new gen03 best)

**Next:** Try higher teacher_ratio_start (0.5 or 0.6) or slower decay schedule. Alternatively, add velocity-norm regularization or higher dropout (0.15→0.2) to push square fold further. Another option: increase model capacity (n_hidden_states 192→256).

---

## Iteration 7 — Higher Teacher Ratio Start (0.5)

**Commit:** d9779f4

**What:** Same as iter6 but teacher_ratio_start=0.5 (from 0.3). Slower approach to free-running rollout.

**Why:** More teacher forcing might give the model more stable gradients early in training, potentially improving convergence.

**Result:** val_MAE=0.3159 (worse than iter6 0.3096). Higher initial teacher ratio hurt. More teacher forcing during training makes the model more dependent on ground truth input and less robust to its own prediction errors at inference.

**Status:** discard — reset to a7b54a4

**Next:** Try larger model capacity: n_hidden_states=256 (from 192). Keep teacher forcing at 0.3. Hypothesis: more hidden capacity in the LSTM could capture more complex quadrotor dynamics, particularly for the square fold.

---

## Iteration 8 — Larger Hidden States (256)

**Commit:** dd5bb50

**What:** n_hidden_states=256 (from 192), all else equal to iter6.

**Why:** More LSTM hidden capacity to capture complex quadrotor dynamics.

**Result:** val_MAE=0.3077 (chirp=0.147, random=0.197, square=0.579). New best, beating iter6 (0.3096). Square improved to 0.579 (from 0.587). Test ensemble MAE=0.093004. Median best epoch=25 suggests fast convergence but possible under-training.

**Status:** keep (new gen03 best)

**Next:** Try n_hidden_states=320 to continue scaling up. Or try 3 LSTM layers (num_layers=3) for more depth. Also consider expanding init network hidden_sizes=[192, 128] to match the larger LSTM.

---

## Iteration 9 — Larger Model (n_hidden=320, wider init)

**Commit:** 82420bd

**What:** n_hidden_states=320 (from 256), hidden_sizes=[192,128] (from [128,64]).

**Why:** Continue scaling model capacity after iter8 improvement.

**Result:** val_MAE=0.3191 (worse than iter8 0.3077). Larger model underperforms within the 300s/fold budget — convergence too slow or overfitting. Median best epoch=25 again (very early).

**Status:** discard — reset to dd5bb50

**Next:** Try deeper LSTM: num_layers=3 with n_hidden=256. More depth may capture temporal dependencies better than width, while staying within the parameter budget.

---

## Iteration 10 — Deeper LSTM (3 layers)

**Commit:** e80fb5c

**What:** num_layers=3 (from 2), n_hidden_states=256.

**Why:** Deeper LSTM can capture longer temporal dependencies. More depth vs width within same parameter budget.

**Result:** val_MAE=0.2958 (chirp=0.142, random=0.180, square=0.566). New best. All folds improved significantly: square 0.566 (from 0.579), random 0.180 (from 0.197). Test ensemble MAE=0.088806. Depth helps more than width alone.

**Status:** keep (new gen03 best)

**Next:** Try num_layers=4 to continue depth scaling. Or try n_hidden=256, 3 layers with lower dropout (0.05) since the model seems to generalize well with depth. Also consider batch_size=128 for more stable gradients with the larger model.

---

## Iteration 11 — 4-Layer LSTM

**Commit:** c985f75

**What:** num_layers=4 (from 3), n_hidden_states=256.

**Why:** Continue depth scaling after iter10's success.

**Result:** val_MAE=0.2953 (chirp=0.140, random=0.179, square=0.567). Marginal new best (0.2953 vs 0.2958). Square essentially unchanged (0.567 vs 0.566). Test ensemble MAE=0.0846. Diminishing returns on depth.

**Status:** keep (new gen03 best, marginal)

**Next:** Try num_layers=5 to see if depth scaling continues. If not, shift to other improvements: lower dropout (0.05), lower weight_decay, or data augmentation changes.

---

## Iteration 12 — 5-Layer LSTM

**Commit:** d463c2e

**What:** num_layers=5 (from 4), n_hidden_states=256.

**Why:** Continue depth scaling after marginal iter11 improvement.

**Result:** val_MAE=0.2986 (worse than iter11 0.2953). 5 layers too deep — within 300s/fold budget, gradients vanish or training doesn't converge as well.

**Status:** discard — reset to c985f75

**Next:** Add state noise to y_prev during rollout (state_noise_std). During training, inject small Gaussian noise into the rollout state y_prev at each step. Forces model to be robust to accumulated state prediction errors, which is the core of the distribution shift (train: teacher forcing; test: free running).

---

## Iteration 13 — Rollout State Noise

**Commit:** f444da0

**What:** Added state_noise_std=0.02 to y_prev at each rollout step during training (only when self.training=True). Forces model to be robust to accumulated prediction errors.

**Why:** Bridge train/test gap for autoregressive rollout — add noise to simulate what happens during inference as errors accumulate.

**Result:** val_MAE=0.3010 (worse than iter11 0.2953). State noise on top of teacher forcing makes the training problem harder without benefit. The two augmentations may conflict: teacher forcing replaces y_prev with truth, while state noise perturbs it — sending mixed signals about what the "true" previous state should look like.

**Status:** discard — reset to c985f75

**Next:** Try GRU instead of LSTM (same num_layers=4, n_hidden=256). GRU has fewer parameters (no cell state, fewer gates) which may lead to faster/better convergence within the 300s time budget. Median best_epoch=25 suggests training converges very quickly; GRU might find a better optimum in the same wall-clock time.

---

## Iteration 14 — GRU instead of LSTM

**Commit:** 670bb0b

**What:** type="GRU" (from LSTM), num_layers=4, n_hidden=256.

**Why:** GRU has fewer parameters per layer (no cell state). Might converge faster within 300s/fold budget.

**Result:** val_MAE=0.3155 (worse than iter11 0.2953). GRU inferior to LSTM despite fewer parameters. The LSTM cell state is beneficial for capturing longer-horizon quadrotor dynamics.

**Status:** discard — reset to c985f75

**Next:** Faster teacher forcing decay: teacher_ratio_start=0.3 decaying to 0 over 100 epochs (not 500). After epoch 100, training runs in full free-running mode matching inference. Hypothesis: spending more of the training budget in free-running mode gives better inference-time generalization.

---

## Iteration 15 — Faster Teacher Decay (100 epochs)

**Commit:** afdd8e5

**What:** teacher_decay_epochs=100 so teacher_ratio decays 0.3→0 over 100 epochs (not 500). All remaining epochs train in free-running mode.

**Why:** If the model performs best at epoch 25 and free-running mode is what we need at inference, spending more training budget in free-running mode might help.

**Result:** val_MAE=0.3021 (worse than iter11 0.2953). Faster decay didn't help; the teacher forcing schedule over all epochs was working better.

**Status:** discard — reset to c985f75

**Next:** Aggressive LR decay: cosine T_max=100 so LR decays from 1e-3 to 1e-5 by epoch 100 (then stays at 1e-5). Fine-tuning at low LR for the remaining epochs may help. With T_max=500, LR is still ~1e-3 at epoch 100 — never truly fine-tuning.

---

## Iteration 16 — Aggressive LR Decay (cosine T_max=100)

**Commit:** aba7ec0

**What:** CosineAnnealingLR with T_max=100 (from 500). LR decays 1e-3→1e-5 by epoch 100, then stays at 1e-5 for remaining epochs.

**Why:** Median best_epoch=25 suggests the model converges quickly. Faster LR decay may allow better fine-tuning after the initial fast convergence.

**Result:** val_MAE=0.2978 (just missed iter11 best of 0.2953). Not a new best. The aggressive LR schedule didn't help.

**Status:** discard — reset to c985f75

**Next:** Add previous motor command [u_prev, u_prev^2] as additional LSTM input. Currently LSTM sees [u_t, u_t^2, y_prev] (20D); adding u_prev gives [u_t, u_t^2, u_prev, u_prev^2, y_prev] (28D). Motor acceleration (u_t - u_prev) should help detect turns in the square fold.

---

## Iteration 17 — Previous Motor Command as LSTM Input

**Commit:** 4ddbcb9

**What:** Added [u_prev, u_prev^2] (8D) to LSTM inputs. LSTM now sees [u_t, u_t^2, u_prev, u_prev^2, y_prev] = 28D. u_prev=zeros at t=0.

**Why:** Motor acceleration (u_t - u_prev) should signal corners/turns in the square fold. More informative input.

**Result:** val_MAE=0.3041 (worse than iter11 0.2953). More input features didn't help — the LSTM already gets y_prev which contains velocity/angular rate that implicitly encodes motor history. Extra inputs add noise.

**Status:** discard — reset to c985f75

**Next:** Try lower dropout (0.05 from 0.1). Model may be over-regularized — relaxing dropout could allow better fitting of the training dynamics.

---

## Iteration 18 — Lower Dropout (0.05)

**Commit:** ef7a9fc

**What:** dropout_prob=0.05 (from 0.1).

**Why:** Model may be over-regularized; less dropout might allow better fitting.

**Result:** val_MAE=0.3005 (worse than iter11 0.2953). Lower dropout didn't help — 0.1 is better calibrated for this dataset size.

**Status:** discard — reset to c985f75

**Next:** Finer evaluation resolution: eval_every=5 (from 25). With eval_every=25 and best_epoch=25 consistently, the true best epoch might be anywhere in 1-25. Finer resolution allows better early stopping and may reveal that the model peaks earlier and then overfits.

---

## Iteration 19 — Finer Evaluation Resolution (eval_every=5)

**Commit:** c94c5fd

**What:** eval_every_override=5 in config_pars, overriding prepare.py's default of 25. Evaluations now happen at epochs 5, 10, 15, 20, 25, 30, ... instead of 25, 50, 75...

**Why:** With eval_every=25 and best_epoch always reported as 25, the true optimal epoch could be anywhere from 1-25. Finer resolution catches the real peak and triggers early stopping sooner (or later at the right time).

**Result:** val_MAE=0.2887 (chirp=0.135, random=0.172, square=0.559). New best by large margin (0.2887 vs 0.2953). Best epochs: chirp=30, random=20, square=45 — all well within 25-50 range. Finer resolution correctly identified non-trivial optimal epochs. Test ensemble MAE=0.084229.

**Status:** keep (new gen03 best)

**Next:** Try eval_every=3 for even finer resolution. Or push model capacity further: n_hidden=288 (between 256 and 320). Also try combining with lower weight_decay (5e-5).

---

## Iteration 20 — Even Finer Evaluation (eval_every=3)

**Commit:** 0cf3bb2

**What:** eval_every_override=3 (from 5).

**Why:** Even finer resolution may catch the exact best epoch.

**Result:** val_MAE=0.2889 (essentially same as iter19 0.2887). No benefit from eval_every=3 vs 5. The resolution is already sufficient at 5-epoch intervals.

**Status:** discard — reset to c94c5fd

**Next:** Retest n_hidden=320 with the finer eval_every=5 (previously tried with eval_every=25 and failed). The coarser eval was masking its potential.

---

## Iteration 21 — Wider LSTM (n_hidden=320) with eval_every=5

**Commit:** 67a9296

**What:** n_hidden_states=320, eval_every=5.

**Why:** Finer eval fixed the n_hidden=256 issue; maybe it also fixes the n_hidden=320 overfitting observed with eval_every=25.

**Result:** val_MAE=0.2911 (worse than iter19 0.2887). n_hidden=320 is genuinely too wide for this dataset, not just a coarse-eval artifact. Optimal capacity is 256.

**Status:** discard — reset to c94c5fd

**Next:** Try weight_decay=5e-5 (half of current 1e-4). Lower regularization may allow better fitting of the dynamics.

---

## Iteration 22 — Lower Weight Decay (5e-5)

**Commit:** f12c49c

**What:** weight_decay=5e-5 (from 1e-4).

**Why:** Less L2 regularization may allow better fitting.

**Result:** val_MAE=0.2911 (worse than iter19 0.2887). Lower weight decay hurts slightly. The 1e-4 regularization is well-calibrated.

**Status:** discard — reset to c94c5fd

**Next:** Extend early stopping patience from 20 to 30 evaluations (with eval_every=5: 20→100 vs 30→150 epochs of patience). May allow the model to escape local minima and find better optima with more training.

---

## Iteration 23 — Longer Early Stopping Patience (30)

**Commit:** e012e05

**What:** early_stopping_patience=30 (from 20) with eval_every=5.

**Why:** With patience=20 and eval_every=5, stopping after 100 epochs of no improvement. Extending to 150 might allow exploring further.

**Result:** val_MAE=0.2887 (same as iter19). Best checkpoint is identical — more patience doesn't help when the model genuinely peaks and then deteriorates.

**Status:** discard — reset to c94c5fd

**Next:** LSTM internal dropout: add dropout=0.1 between LSTM layers (PyTorch nn.LSTM dropout param). With 4 layers, this regularizes between layers 1-2, 2-3, 3-4. May improve generalization.

---

## Iteration 24 — LSTM Internal Dropout (0.1 between layers)

**Commit:** c824f40

**What:** lstm_dropout=0.1 applied between LSTM layers (PyTorch dropout param, applied to layers 1-2, 2-3, 3-4).

**Why:** Additional regularization between LSTM layers may reduce inter-layer overfitting.

**Result:** val_MAE=0.2899 (slightly worse than iter19 0.2887). LSTM internal dropout adds regularization but hurts performance slightly — the 4-layer LSTM already has enough implicit regularization from batch training.

**Status:** discard — reset to c94c5fd

**Next:** Add step-1 term to multihorizon loss: loss = loss_50 + 0.5*loss_25 + 0.25*loss_10 + 0.5*loss_1. Penalizing single-step errors forces the model to get immediate dynamics right, which should help with error accumulation in the square fold.

---

## Iteration 25 — Step-1 Multihorizon Loss Term

**Commit:** 36a3ec0

**What:** Added 0.5*loss_1 to multihorizon loss: total = loss_50 + 0.5*loss_25 + 0.25*loss_10 + 0.5*loss_1.

**Why:** Heavily penalizing single-step errors forces model to get immediate dynamics right, reducing first-step error that compounds over the rollout (especially in square fold).

**Result:** val_MAE=0.2896 (marginally worse than iter19 0.2887). The step-1 penalty adds complexity to the loss without clear benefit — the existing multihorizon already covers short-horizon errors reasonably.

**Status:** discard — reset to c94c5fd

**Next:** Try larger batch size (128 from 64) for more stable gradient estimates. With teacher forcing and complex 4-layer LSTM, larger batches may reduce noise in the gradient direction.

---

## Iteration 26 — Larger Batch Size (128)

**Commit:** 512f354

**What:** batch_size=128 (from 64), all else equal to iter19.

**Why:** More stable gradient estimates with larger batches, especially helpful for the complex 4-layer LSTM with teacher forcing.

**Result:** val_MAE=0.2878 (chirp=0.129, random=0.162, square=0.572). New best. Chirp (0.129 vs 0.135) and random (0.162 vs 0.172) improved significantly. Square slightly worse (0.572 vs 0.559). Test ensemble MAE=0.082455. Larger batches stabilize training.

**Status:** keep (new gen03 best)

**Next:** Try batch_size=256 to continue batch scaling. Or try combined: batch_size=128 + n_hidden=256 + 5 layers (previously 5 layers was worse with batch=64; might work with batch=128).

---

## Iteration 27 — Batch Size 256

**Commit:** 93e4eff

**What:** batch_size=256 (from 128).

**Why:** Continue batch scaling after iter26 improvement.

**Result:** val_MAE=0.2882 (marginally worse than iter26 0.2878). Essentially tied — 256 slightly over-batched for the dataset size, reducing gradient noise benefit.

**Status:** discard — reset to 512f354

**Next:** Try num_layers=5 with batch_size=128. Previously num_layers=5 with batch=64 failed (0.299), but larger batches may stabilize the deeper network training.

---

## Iteration 28 — 5-Layer LSTM with batch=128

**Commit:** 8e376ee

**What:** num_layers=5 + batch_size=128. Previously num_layers=5 with batch=64 gave 0.299; now with batch=128.

**Why:** Larger batches stabilize training for deeper networks. 5-layer LSTM was failing with noisy small-batch gradients.

**Result:** val_MAE=0.2858 (chirp=0.125, random=0.162, square=0.571). New best. Chirp 0.125 (from 0.129). Square slightly improved (0.571 vs 0.572). Test ensemble MAE=0.081634. Depth + stable batches is the winning combination.

**Status:** keep (new gen03 best)

**Next:** Try num_layers=6 with batch=128. Or try num_layers=5 + batch=256. Or try n_hidden=320 with 5 layers.

---

## Iteration 29 — 6-Layer LSTM with batch=128

**Commit:** 2184c66

**What:** num_layers=6 + batch_size=128.

**Why:** Continue depth scaling after iter28 improvement.

**Result:** val_MAE=0.2984 (worse than iter28 0.2858). 6 layers too deep — even with batch=128, gradient vanishing/instability is a problem. Optimal depth with batch=128 is 5 layers.

**Status:** discard — reset to 8e376ee

**Next:** Try n_hidden=320 with 5 layers + batch=128. Previously n_hidden=320 with 4 layers and eval_every=25 failed; with 5 layers and finer eval/larger batch this might work.

---

## Iteration 30 — n_hidden=320, 5 layers, batch=128

**Commit:** b81e9b0

**What:** n_hidden=320 (from 256), 5 layers, batch=128.

**Why:** 256-hidden 5-layer model is the current best; try wider hidden state for more capacity.

**Result:** val_MAE=0.2890 (worse than iter28 0.2858). n_hidden=320 with 5 layers over-parameterized — over-fitting or harder optimization landscape. n_hidden=256 remains optimal.

**Status:** discard — reset to 8e376ee

**Next:** Try teacher_ratio_start=0.2 (lower than current 0.3). Less teacher forcing may improve rollout generalization, especially on the square fold.

---

