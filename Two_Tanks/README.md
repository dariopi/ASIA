# Cascaded Tank Benchmark

Benchmark and training code for ASIA on the cascaded tank system.

## Project Overview

The project trains a neural sequence model to predict the tank output from the input trajectory using the official benchmark data.

- Training data: one full training trajectory split into 2 contiguous folds
- Test data: one official test trajectory
- Initial conditions: the previous 5 input/output samples

## Main Files

- [prepare.py](./prepare.py): dataset creation, cached data, contiguous fold generation, initial-condition construction
- [model.py](./model.py): neural model definition
- [train.py](./train.py): cross-validation training used during the autoresearch loop
- [test.py](./test.py): denormalized ensemble test evaluation using the saved fold checkpoints
- [PROGRAM.md](./PROGRAM.md): instructions for the autonomous experimentation loop
- [Cascaded_description.md](./Cascaded_description.md): short benchmark description

## Repository Branches

The repository has three branches:

- `main`: baseline starting point
- `best`: best-performing model found during the ASIA workflow
- `final`: final state after the AI agent completed the experimentation cycle

To reproduce the best model:

```bash
git checkout best
uv sync
uv run train.py
```

To inspect the final state:

```bash
git checkout final
```
