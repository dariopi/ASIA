# Cascaded Two-Tank System Benchmark

## 1. System Description

The cascaded two-tank system is a nonlinear dynamical system consisting of:

- Two water tanks connected in series  
- A pump feeding the upper tank  
- Two outlets regulating the flow between tanks and back to the reservoir  

### Structure

u(t) → pump → Tank 1 → Tank 2 → reservoir
                         ↓
                       y(t)

- Input: pump voltage u(t)  
- States:
  - x1(t): water level in Tank 1  
  - x2(t): water level in Tank 2  
- Output: y(t) = x2(t)

---

## 2. Physical Modeling (Grey-box)

The system can be approximated using:

- Mass conservation
- Bernoulli’s principle (Torricelli’s law for outflow)

### Continuous-time model

dx1/dt = -k1 * sqrt(x1) + k4 * u(t) + w1(t)

dx2/dt = k2 * sqrt(x1) - k3 * sqrt(x2) + w2(t)

y(t) = x2(t) + e(t)

### Interpretation

- sqrt(x): nonlinear outflow due to fluid dynamics  
- k1, k2, k3, k4: physical parameters  
- w1(t), w2(t): process noise  
- e(t): measurement noise  

---

## 3. Nonlinearities

### Soft nonlinearities
- Origin: square-root flow terms sqrt(x)

### Hard nonlinearities
- Origin: overflow saturation
- Occurs when Tank 1 reaches its maximum capacity

---

## 4. Overflow Effect

When the input u(t) is large:

- Tank 1 overflows
- The flow splits into:
  - flow to Tank 2
  - direct flow to the reservoir

### Consequences

- Input-dependent process noise
- Non-deterministic behavior
- Nonlinear distortion

---

## 5. Input-Output Trajectories

### Input signal u(t)

- Type: multisine  
- Frequency range: 0 – 0.0144 Hz  
- Sampling time: Ts = 4 s  

### Output signal y(t)

- Measured water level in Tank 2  

Characteristics:

- Smooth response  
- Delayed relative to input  
- Nonlinear distortions  
- Saturation effects