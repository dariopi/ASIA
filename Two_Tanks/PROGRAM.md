# Cascaded Tank Benchmark Program

This folder contains the material used to run autoresearch on the cascaded tank benchmark. The goal of the project is to train and evaluate neural models that describe the system dynamics from input-output data.

## Benchmark Description

The benchmark is described in [Cascaded_description.md](./Cascaded_description.md).

## Folder Structure

The main files are:

- `prepare.py`
- `model.py`
- `train.py`
- `test.py`
- `Cascaded_description.md`

Below is the role of each file.

## `prepare.py`

The file `prepare.py` is responsible for:

- loading the official benchmark data
- splitting the full training trajectory into 3 contiguous folds used for CV
- building the initial-condition vectors from the previous 5 samples
- saving preprocessed data in cache

This file also contains general configuration parameters that **should not be changed**. In particular, it defines how the data are read, how they are split into folds, how initial conditions are built, and which cached files are produced.

In short, `prepare.py` prepares everything needed before training.

## `model.py`

The file `model.py` contains the model architectures used to describe the benchmark dynamics.

This file **can be modified**. It is the correct place to work if you want to change or improve the model architecture. For instance, you may use different recurrent models, state-space models, physical models, or hybrid models with learned residual terms.

## `train.py`

The file `train.py` runs the neural network training.

At the beginning of the file there is a dictionary called `config_pars`, which contains the main training hyperparameters, for example:

- learning rate
- recurrent network type
- number of epochs
- number of hidden states
- hidden layer sizes
- activation function
- number of layers
- dropout
- direct feedthrough

These parameters **can be modified**. They are one possible experimentation space, but they are not privileged over `model.py`: the agent is free to change the model architecture, the training setup, or both, depending on what seems most promising.

The training loop must still respect the fixed benchmark settings coming from `prepare.py`, including `eval_every` and `fold_time_budget_seconds`.

The choice of hyperparameters should be carried out using 3-fold cross-validation. Each validation fold corresponds to one contiguous sequence extracted from the full training trajectory. The metric used to select the hyperparameters is the aggregated RMSE across the folds.

Read the current version of `train.py` to better understand how the file is organized and which metrics are used.

## `test.py`

The file `test.py` evaluates the saved fold models on the official test trajectory. It cannot be used by autoresearch. It is only to assess final performance on test data.

## Training Log

During training, the code writes one log file for each validation fold.

The training includes a **3-fold cross-validation** phase. In this phase:

- at each fold, one sequence is used for validation
- the other sequences are used for training
- the log shows the evolution of the training loss and of the training/validation RMSE

The time budget for each fold is fixed at **fold_time_budget_seconds seconds**, with fold_time_budget_seconds specified in prepare.py. If a fold reaches that budget, the training loop should stop for that fold, keep the best validation checkpoint reached, and move on to the next one. Hitting the budget is not a fail in training. The fold time budget is a normal stopping condition, not an error.


## Typical Workflow

To use this folder correctly, the recommended workflow is:

1. Read `Cascaded_description.md` to understand the benchmark.
2. Run `prepare.py` to load and preprocess the datasets. This file should not be modified.
3. Modify `model.py` and/or `train.py` depending on the experimental idea you want to test.
4. In `train.py`, update `config_pars` or other training logic if that is the most useful change for the current candidate.
5. Run `train.py` to train the model and inspect the fold logs.
6. Run `test.py` only after the search decision has been made.

## Setting Up a New Experiment

To set up a new experiment, work with the user through the following steps.

### 1. Agree on a Run Tag

Before starting a new run, agree on a run tag with the user. A simple convention is to use a tag based on the current date or a short progressive label, for example `gen01`.

The branch name should have the form:

- `autoresearch/<tag>`

This branch must not already exist. Each new experiment should start from a fresh branch.

### 2. Create the Branch

Create the new branch starting from the current main branch:

```bash
git checkout -b autoresearch/<tag>
```

This ensures that the experiment starts from a clean and well-defined base.

### 3. Read the In-Scope Files

Before starting, read the main files involved in the experiment.

The repository is small, so the important files should be reviewed completely:

- `Cascaded_description.md`
- `prepare.py`(cannot be modified)
- `model.py` (can be modified)
- `train.py` (can be modified)

### 4. Initialize `results.tsv`

At the beginning, create a file called `results.tsv` containing only the header row.

At the same time, `search_journal.md` must also be reset for the new experiment: it should either not exist yet or contain no iteration entries from previous experiments. Its iteration count must start again from `0` for the new branch, just like `results.tsv` starts fresh.

The baseline result should be recorded after the first run.

### 5. Confirm the Setup

Before launching the first run, confirm that:

- the run tag has been agreed with the user
- the branch `autoresearch/<tag>` has been created
- the relevant files have been read
- `results.tsv` has been initialized
- `search_journal.md` has been reset for the new experiment and will start again from iteration `0`

Once the setup is confirmed, experimentation can begin.

## Experiment Rules

The training script is launched as:

```bash
python train.py
```

In this project, the main objective is to improve the **3-fold cross-validation performance**. The reference metric to optimize is the **aggregated RMSE across the 3 validation folds**.

### What You Can Modify

Only the following files should be edited during experiments:

- `model.py`
- `train.py`

In these files, you are free to modify:

- model architecture
- model type
- optimizer
- training hyperparameters
- training loop
- model size
- any other training-related choice implemented inside these two files

### What You Cannot Modify

The following constraints should be respected:

- `prepare.py` must not be modified. It should be treated as read-only.
- no new packages or dependencies should be installed
- only the packages already available in the current project may be used
- the evaluation protocol must not be changed

### Target Metric

The goal is simple:

- obtain the lowest possible **aggregated RMSE on 3-fold cross-validation**

This is the main metric that should guide hyperparameter selection and model changes.

The test trajectory must not be used to decide which model to keep.

### Runtime and Practical Constraints

The training code must:

- run without crashing
- complete correctly on the available hardware
- remain compatible with the current project setup


## Search Strategy

The search should prioritize changes that have a realistic chance of producing substantial improvements.

Do not spend too much time on minor parameter nudges such as:

- slightly changing `weight_decay`
- changing the width of a hidden layer by a very small amount
- other very small hyperparameter adjustments that are unlikely to move the metric meaningfully

These minor refinements are still allowed, but they should be used only occasionally, for example after a genuinely promising model family has already been identified and you want to refine it.

As a general strategy:

1. In the first 6–7 runs, explore meaningfully different model architectures and training objectives.
2. When a promising architecture is found, spend a few runs refining it through moderate hyperparameter or training-loop changes.
3. If these refinements do not produce meaningful validation-RMSE improvements, stop refining that architecture and try a different model family or substantially different modeling idea.
4. Repeat this explore/refine cycle: explore broad alternatives, then refine only the candidates that show clear promise.

The agent should therefore be willing to test:

- different neural architectures
- different recurrent model families
- different state-initialization mechanisms
- residual or direct-feedthrough variants
- physics-inspired or partially physical models
- other materially different modeling choices


### First Run

The first run of a new experiment should always be the baseline run.

This means that the training script should first be executed in its current form, without modifying the starting configuration, so that future changes can be compared against a clear reference result.

For this baseline run, neither `model.py` nor `train.py` should be changed. The baseline must measure the branch starting point exactly as it is.

## Logging Results

When an experiment is completed, record it in `results.tsv`.

This file should be **tab-separated**, not comma-separated.

The file should contain a header row and the following columns:

```text
commit	val_RMSE	status	description
```

The columns mean:

1. `commit`: the short Git commit hash, typically 7 characters
2. `val_RMSE`: the aggregated RMSE obtained on 3-fold cross-validation
3. `status`: typically `keep` or `discard`
4. `description`: a short explanation of what the experiment changed

> **IMPORTANT — test set is off-limits during search.**
> The only metric recorded in `results.tsv` is `val_RMSE`.
> Do **NOT** add a `test_RMSE` column. Do **NOT** run `test.py` at any point during the experiment loop.
> The test set must remain unseen until the very final evaluation after all experiments are complete.

If a run crashes and no valid result is produced, you may record:

- `inf` as the RMSE value
- `discard` as the status
- a short crash description in the text field

Example:

```text
commit	val_RMSE	status	description
a1b2c3d	0.997900	keep	baseline
b2c3d4e	0.993200	keep	increase learning rate
c3d4e5f	1.005000	discard	switch from LSTM to GRU
d4e5f6g	inf	discard	crash due to shape mismatch
```

`results.tsv` is meant to be a local experiment log and should remain untracked by Git.

## Experiment Loop

Experiments are expected to run on a dedicated branch, for example:

- `autoresearch/gen01`
- `autoresearch/gen01-a`

The typical experiment loop is:

1. Check the current Git state, including the active branch and current commit.
2. If this is the first run of a brand-new experiment, run the baseline exactly as-is, without modifying `model.py` or `train.py`.
3. For every later run, modify `train.py` and/or `model.py` with one experimental idea.
4. Create a Git commit for that experiment when the run is not the untouched baseline.
5. Run the experiment using:

```bash
python train.py
```

6. Inspect the latest fold logs under `logs/`.
7. Read the final validation RMSE summary printed by `train.py`.
8. Decide whether the experiment should be marked as `keep` or `discard`.
9. Update `results.tsv` with the validation RMSE and status.
10. Update `search_journal.md` with an explicit note describing what was changed, why that change was made, what happened, and what should be tried next.
11. If the run is marked `discard`, return the repository to the current best kept commit before starting the next experiment.
12. Continue with the next experiment unless a real blocker requires a decision from the user.


## Commit Rules

Each non-baseline experiment should be committed after the code change has been made and before the run result is recorded.

If a run is marked `keep`, the corresponding model code must exist as a Git commit. In practice, every kept non-baseline run should already have its own commit before the run is logged.

If a run is marked `discard`, the search must not continue from that discarded code state. Before starting the next candidate, reset the repository to the best kept commit currently known for the branch.

The commit message should be short and descriptive, for example:

```text
try deeper init network
```

or:

```text
switch recurrent block to GRU
```

The baseline run may remain uncommitted if it corresponds exactly to the starting state of the branch.

The search should always have a clear notion of the current best kept commit. New experiments should branch logically from that best kept point, not from the latest discarded attempt.

## Search Journal Rules

`search_journal.md` should be explanatory, not just a terse ledger. After every completed run, write short but explicit sentences that state:

- what was changed in `model.py` and/or `train.py`
- why that change was chosen, meaning the hypothesis behind it
- what happened in the run, including the important metric or training behavior
- whether the run was kept or discarded
- what the next step should be

The journal should make it easy for a reader to understand the reasoning of the search loop without having to infer it from Git diffs alone.

## AI Agent Rules

If an AI agent is launched to run the experiment loop, it should follow these rules:

- read `PROGRAM.md` before starting
- treat `prepare.py` as read-only
- modify only `model.py` and `train.py`
- keep `train.py` as a single-run trainer
- do not create loop scripts inside the repository
- use the fold logs in `logs/` to inspect training behavior
- use validation RMSE, not test RMSE, to decide what to keep
- update `results.tsv` after each completed run
- update `search_journal.md` after each completed run
- create Git commits for non-baseline experiments
- ensure every kept non-baseline run corresponds to a Git commit
- after every discarded run, reset back to the best kept commit before continuing
- continue autonomously unless a real blocker requires user input