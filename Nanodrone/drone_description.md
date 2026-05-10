## Benchmark

### Model

- **Platform:** Crazyflie 2.1 Brushless nano-quadrotor.
- **Nominal physical model:** continuous-time rigid-body quadrotor dynamics with quadratic aerodynamic forces and moments, discretized for multi-step prediction.
- **State:** 
  \[
  x = [p^\top, v^\top, q^\top, \omega^\top]^\top
  \]
  where \(p\) is position, \(v\) linear velocity, \(q\) quaternion orientation, and \(\omega\) angular velocity.
- **Inputs:** rotational speeds of the four propellers.
- **Outputs:** position, attitude (roll, pitch, yaw), linear velocities, and angular velocities.


---

### Dataset

- **Type:** real-world flight data from aggressive maneuvers.
- **Size:** ~75,000 samples.
- **Trajectories:** four trajectories (Square, Random, Chirp, Melon).
- **Inputs:** four motor speeds (control inputs).
- **Outputs:** position, attitude (roll, pitch, yaw), linear velocities, angular velocities.
- **Characteristics:**
  - Strong nonlinearities and coupling effects  
  - Suitable for multi-step prediction and system identification  
- **Train/Test split:**
  - Training: Square, Random, Chirp  
  - Test: Melon  
- **Training setup:** fixed-length windows with prediction horizon \(H = 50\) (≈0.5 s).

---

### Baselines

1. **Physics-based model**  
   - Derived from nominal quadrotor dynamics and aerodynamic equations.  
   - Simplified rotational dynamics used for better empirical fit.

2. **Feed-forward residual model (MLP)**  
   - Predicts output increments given current state and inputs.  
   - Rolled out recursively over the prediction horizon.

3. **LSTM model**  
   - Recurrent model with learned hidden-state initialization.  
   - Multi-step prediction via recursive rollout.

4. **Hybrid physics + residual model**  
   - Combines nominal physical model with a neural residual correction.  
   - Captures unmodeled effects (e.g., drag, actuator dynamics).
