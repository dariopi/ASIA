# ASIA: Autonomous System Identification Agent

ASIA is an agentic AI framework for system identification. It delegates the iterative search over model classes, architectures, and training strategies to a large language model (LLM) acting as an autonomous coding agent. Given only a plain-English description of the identification problem and a fixed evaluation protocol, at each iteration the agent proposes a new configuration, implements it as executable code, runs training, observes the cross-validation metric, and decides the next move. The framework is built on the [autoresearch](https://github.com/karpathy/autoresearch) platform and is described in the paper:

> D. Piga and M. Forgione, *ASIA: an AI Agent for System Identification*, 2026.

# Repository

This repository contains two ASIA benchmark projects:

- [Two_Tanks](./Two_Tanks): ASIA for the cascaded two-tank benchmark
- [Nanodrone](./Nanodrone): ASIA for the nano-drone benchmark

Each subfolder is a self-contained benchmark project with its own dataset preparation, model definition, training loop, evaluation script, and project-specific documentation.

## Repository Organization

At a high level, each benchmark subfolder follows the same structure:

- `prepare.py`: prepares the benchmark data, cached files, and fold splits
- `model.py`: defines the model architecture
- `train.py`: runs the training and cross-validation loop used by ASIA
- `test.py`: evaluates the saved checkpoints on the official test data
- `PROGRAM.md`: contains the instructions for the autonomous experimentation loop
- `README.md`: contains benchmark-specific details
- `pyproject.toml` and `uv.lock`: define and lock the Python environment for that specific project

## Repository History

The repository has three tagged milestones on the `main` branch:

- `main`: baseline starting point, before any autonomous experimentation
- `best`: best-performing model found during the ASIA workflow
- `final`: final state after the AI agent completed the experimentation cycle

To inspect a specific milestone:

```bash
git checkout main   # baseline
git checkout best   # best model
git checkout final  # final experiments
```

To return to the latest state:

```bash
git switch main
```

## Available Projects

### Cascaded Tank System

The [Two_Tanks](./Two_Tanks) project applies ASIA to the cascaded tank benchmark.

Reference benchmark paper:

- M. Schoukens, P. Mattsson, T. Wigren, and J. M. M. G. Noel, [*Cascaded Tanks Benchmark Combining Soft and Hard Nonlinearities*](https://research.tue.nl/en/publications/cascaded-tanks-benchmark-combining-soft-and-hard-nonlinearities)
- M. Schoukens and J. P. Noel, [*Three Benchmarks Addressing Open Challenges in Nonlinear System Identification*](https://www.sciencedirect.com/science/article/pii/S2405896317300915)

### Nanodrone Benchmark

The [Nanodrone](./Nanodrone) project applies ASIA to the nano-drone benchmark.

Reference benchmark paper:

- Riccardo Busetto, Elia Cereda, Marco Forgione, Gabriele Maroni, Dario Piga, and Daniele Palossi, [*Nonlinear system identification for a nano-drone benchmark*](https://doi.org/10.1016/j.conengprac.2026.106871)

## Typical Workflow

The recommended workflow is the same for both projects.

1. Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Move into the benchmark folder you want to run:

```bash
cd Two_Tanks
```

or:

```bash
cd Nanodrone
```

3. Sync the project environment:

```bash
uv sync
```

4. Prepare the cached data, folds, and evaluation metadata:

```bash
uv run prepare.py
```

5. Run a one-shot training pass to verify that the setup is working:

```bash
uv run train.py
```

6. Specify your autonomous experimentation loop in plain English in the local `PROGRAM.md`.

7. Use your AI agent (e.g., Claude or Codex) in the selected project folder. Disable all permissions. Then start the autonomous loop using the (suggested) prompt in [How to Call the AI Agent](#how-to-call-the-ai-agent), modifying them if necessary for the specific problem.

During this phase, the agent modifies only:
- `model.py`
- `train.py`

8. When ASIA loop is finished, evaluate the saved fold checkpoints on the official test data:

```bash
uv run test.py
```

To evaluate the archived best-so-far snapshot instead of the most recent run:

```bash
uv run test.py --checkpoint-set best_so_far
```

9. Review the main ASIA outputs:

- `results.tsv`: records the validation results and the architectures or modeling choices that were tried across iterations
- `search_journal.md`: provides a plain-English explanation of what was done at each iteration, why it was done, and what should be tried next

## How to Call the AI Agent

Suggested prompt:

```text
Have a look at `PROGRAM.md` and start a new autonomous experiment cycle. Do the setup first, then keep going autonomously after each completed run by inspecting the latest logs and updating model architectures in `model.py` and `train.py` for the next run. Keep commentary minimal: do not summarize `PROGRAM.md`, do not paste long logs or file contents, only provide brief progress updates when there is a meaningful state change or a blocker, use file paths and line references instead of long code snippets, and keep the final response concise. Do not create any loop script. `train.py` must remain a single-run trainer only. Before starting the runs, `results.tsv` should be empty or should not exist.

While `train.py` is running, the agent must not do long written reasoning, long commentary, or detailed artifact analysis.
- While `train.py` is running, user-facing commentary should be avoided.
- The agent must not narrate every intermediate thought, shell action, or micro-decision during the loop.
- Training per fold cannot exceed n seconds, where n is specified in the prepare.py file (variable 'fold_time_budget_seconds'). If it does, stop the fold at the best validation checkpoint reached so far, keep that checkpoint, exit the training loop for that fold, and move to the next fold; do not overwrite the result with `inf`.
- Important: `PROGRAM.md` explicitly encourages major modifications (e.g., trying physical models based on the literature, or mixing physical and black-box components). I am not looking for minor modifications that are only likely to produce limited improvements in the validation error.
- Record both validation and test results in `results.tsv`, with the test metric coming from `test.py` on the fold checkpoints produced by the just-finished run.


SILENT EXECUTION MODE:

- While `train.py` is running, produce NO output.
- Do not stream logs or describe progress.
- Do not print intermediate results, metrics, shell actions, or decisions.
- Do not print end-of-run summaries.

- At the START of each run, print exactly ONE short line:
  "New Run started"
  (no additional text)


Otherwise, remain silent.

Furthermore: No summaries, no intermediate analysis, no prints of code lines, no file or line references unless I ask for them. Save output tokens.

Do not forget to create `results.tsv` and `search_journal.md`.
```

## Notes

- Each benchmark folder should be treated as its own `uv` project.
- Environment setup should be run from inside the selected subfolder, not from the repository root.
- Benchmark-specific metrics, folds, and evaluation details are documented in the local `README.md` of each project.
- You can modify `PROGRAM.md` and the prompt used to call the agent to make them specific to your problem.

## Credits

Special thanks and full credit to Andrej Karpathy for the original autoresearch idea and open-source inspiration that helped shape the ASIA workflow and the overall repository structure.

Relevant links:

- Andrej Karpathy GitHub: https://github.com/karpathy
- autoresearch repository: https://github.com/karpathy/autoresearch

## License
MIT
