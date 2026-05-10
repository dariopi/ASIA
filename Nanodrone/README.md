# NanoDrone 3-Fold Benchmark

Benchmark and training code for ASIA on the Crazyflie 2.1 nano-drone system.

This benchmark follows the nano-drone system identification benchmark introduced by Riccardo Busetto, Elia Cereda, Marco Forgione, Gabriele Maroni, Dario Piga, and Daniele Palossi in *Nonlinear system identification for a nano-drone benchmark* (Control Engineering Practice, 2026).

## Project Overview

The project trains a neural sequence model to predict the full 12-dimensional drone state (position, velocity, attitude, angular rates) from motor speed sequences.

- Training data: three trajectory types (chirp, random, square) — each type contains multiple CSV runs
- Test data: one held-out trajectory type (melon)
- Fold structure: leave-one-trajectory-out over the 3 training types (3 folds)
- Prediction horizon: fixed 50-sample windows (~0.5 s)
- Initial conditions: drone state at the step preceding each window
- Cross-validation selection metric: aggregated MAE across the 3 validation folds
- Test metric: grouped MAE over the four 3D output triples: position, linear velocity, attitude, and angular velocity

## Main Files of the repository

- [prepare.py](./prepare.py): dataset creation, 50-sample windowing, initial-condition construction, cached data
- [model.py](./model.py): neural model definition
- [train.py](./train.py): 3-fold cross-validation training used during the autoresearch loop
- [test.py](./test.py): denormalized ensemble test evaluation
- [PROGRAM.md](./PROGRAM.md): instructions for the autonomous experimentation loop
- [drone_description.md](./drone_description.md): short benchmark description