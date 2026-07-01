from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"

input_names = ["m1_rads", "m2_rads", "m3_rads", "m4_rads"]
output_names = ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "wx", "wy", "wz"]

config_pars_general = {
    "benchmark_name": "NanoDrone_3Fold",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "num_folds": 3,
    "seq_len": 50,
    "validation_metric": "mae",
    "base_path": "./cached_data",
    "checkpoint_path": "./checkpoints",
    "plots_path": "./plots",
    "log_dir": "./logs",
    "eval_every": 25,
    "fold_time_budget_seconds": 300.0,
    "n_inputs": 4,
    "n_outputs": 12,
    "n_states": 12,
    "train_list": ["chirp", "random", "square"],
    "test_list": ["melon"],
    "warmup_test": 0,
}


@dataclass
class DroneSequence:
    name: str
    u: torch.Tensor   # (N, seq_len, n_inputs)
    y: torch.Tensor   # (N, seq_len, n_outputs)
    y0: torch.Tensor  # (N, 1, n_outputs)  — state at the step preceding each window

    @property
    def num_sequences(self) -> int:
        return int(self.u.shape[0])

    def clone(self) -> "DroneSequence":
        return DroneSequence(
            name=self.name,
            u=self.u.clone(),
            y=self.y.clone(),
            y0=self.y0.clone(),
        )


class Normalizer:
    def __init__(
        self,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        self.u_mean = u_mean.float()
        self.u_std = torch.clamp(u_std.float(), min=eps)
        self.y_mean = y_mean.float()
        self.y_std = torch.clamp(y_std.float(), min=eps)
        self.eps = eps

    @classmethod
    def fit(cls, sequences: list[DroneSequence], eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        u_mean = u_all.mean(dim=0, keepdim=True)
        u_std = u_all.std(dim=0, keepdim=True, unbiased=False)
        y_mean = y_all.mean(dim=0, keepdim=True)
        y_std = y_all.std(dim=0, keepdim=True, unbiased=False)
        return cls(u_mean, u_std, y_mean, y_std, eps=eps)

    def normalize_sequence(self, sequence: DroneSequence) -> DroneSequence:
        normalized = sequence.clone()
        normalized.u = (normalized.u - self.u_mean) / self.u_std
        normalized.y = (normalized.y - self.y_mean) / self.y_std
        normalized.y0 = (normalized.y0 - self.y_mean) / self.y_std
        return normalized

    def denormalize_y_tensor(self, y: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean.to(device=y.device, dtype=y.dtype)
        std = self.y_std.to(device=y.device, dtype=y.dtype)
        return y * std + mean

    def state_dict(self) -> dict:
        return {
            "u_mean": self.u_mean.detach().cpu().numpy(),
            "u_std": self.u_std.detach().cpu().numpy(),
            "y_mean": self.y_mean.detach().cpu().numpy(),
            "y_std": self.y_std.detach().cpu().numpy(),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "Normalizer":
        return cls(
            u_mean=torch.as_tensor(state["u_mean"], dtype=torch.float32),
            u_std=torch.as_tensor(state["u_std"], dtype=torch.float32),
            y_mean=torch.as_tensor(state["y_mean"], dtype=torch.float32),
            y_std=torch.as_tensor(state["y_std"], dtype=torch.float32),
            eps=float(state.get("eps", 1e-6)),
        )


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE averaged over all windows, time steps, and output dimensions."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _load_csvs_for_trajectory(data_dir: Path, trajectory_name: str) -> list[pd.DataFrame]:
    dfs = []
    for fname in sorted(data_dir.iterdir()):
        if fname.name.startswith(trajectory_name) and fname.suffix == ".csv":
            dfs.append(pd.read_csv(fname))
            print(f"    {fname.name}  ({len(dfs[-1])} rows)")
    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found for trajectory '{trajectory_name}' in {data_dir}"
        )
    return dfs


def _windows_from_dataframes(
    dfs: list[pd.DataFrame],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split each DataFrame into non-overlapping windows of length seq_len.
    y0[i] is the drone state at the step immediately before window i.
    """
    u_list, y_list, y0_list = [], [], []

    for df in dfs:
        N = len(df)
        n = int(np.ceil((N - 1) / seq_len))
        u = np.zeros((n, seq_len, len(input_names)), dtype=np.float32)
        y = np.zeros((n, seq_len, len(output_names)), dtype=np.float32)
        y0 = np.zeros((n, 1, len(output_names)), dtype=np.float32)

        for ind in range(n):
            if ind < n - 1:
                u[ind] = df.iloc[ind * seq_len + 1:(ind + 1) * seq_len + 1][input_names].values
                y[ind] = df.iloc[ind * seq_len + 1:(ind + 1) * seq_len + 1][output_names].values
                y0[ind, 0] = df.iloc[ind * seq_len][output_names].values
            else:
                u[ind] = df.iloc[N - seq_len:][input_names].values
                y[ind] = df.iloc[N - seq_len:][output_names].values
                y0[ind, 0] = df.iloc[N - seq_len - 1][output_names].values

        u_list.append(u)
        y_list.append(y)
        y0_list.append(y0)

    return (
        np.concatenate(u_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(y0_list, axis=0),
    )


def load_trajectory(data_dir: Path, name: str, seq_len: int) -> DroneSequence:
    dfs = _load_csvs_for_trajectory(data_dir, name)
    u_np, y_np, y0_np = _windows_from_dataframes(dfs, seq_len)
    return DroneSequence(
        name=name,
        u=torch.from_numpy(u_np),
        y=torch.from_numpy(y_np),
        y0=torch.from_numpy(y0_np),
    )


def save_sequence(path: Path, sequence: DroneSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.asarray(sequence.name),
        u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32),
        y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
    )


def load_sequence(path: Path) -> DroneSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    data = np.load(path, allow_pickle=True)
    name_val = data["name"]
    name = name_val.item() if np.asarray(name_val).shape == () else str(name_val)
    return DroneSequence(
        name=str(name),
        u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
        y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)),
        y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
    )


def save_config(config: dict) -> None:
    base_path = Path(config["base_path"])
    base_path.mkdir(parents=True, exist_ok=True)
    with (base_path / "config_params_general.pkl").open("wb") as handle:
        pickle.dump(config, handle)


def load_config(base_path: str | Path | None = None) -> dict:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    path = root / "config_params_general.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached config: {path}. Run `prepare.py` first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_train_sequences(base_path: str | Path | None = None) -> dict[str, DroneSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(
            f"Missing cached training sequences under {train_dir}. Run `prepare.py` first."
        )
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_test_sequences(base_path: str | Path | None = None) -> dict[str, DroneSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    test_dir = root / "test_sequences"
    sequence_paths = sorted(test_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(
            f"Missing cached test sequences under {test_dir}. Run `prepare.py` first."
        )
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_datasets_and_config(
    base_path: str | Path | None = None,
) -> tuple[dict[str, DroneSequence], dict[str, DroneSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return load_train_sequences(root), load_test_sequences(root), load_config(root)


def load_train_sequences_and_config(
    base_path: str | Path | None = None,
) -> tuple[dict[str, DroneSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return load_train_sequences(root), load_config(root)


def save_metadata(
    config: dict,
    train_sequences: dict[str, DroneSequence],
    test_sequences: dict[str, DroneSequence],
) -> None:
    base_path = Path(config["base_path"])
    metadata = {
        "benchmark_name": config["benchmark_name"],
        "validation_metric": config["validation_metric"],
        "device": config["device"],
        "num_folds": config["num_folds"],
        "seq_len": config["seq_len"],
        "n_inputs": config["n_inputs"],
        "n_outputs": config["n_outputs"],
        "n_states": config["n_states"],
        "input_names": input_names,
        "output_names": output_names,
        "train_sequences": [
            {"name": name, "num_windows": seq.num_sequences}
            for name, seq in train_sequences.items()
        ],
        "test_sequences": [
            {"name": name, "num_windows": seq.num_sequences}
            for name, seq in test_sequences.items()
        ],
        "notes": [
            "Each fold is one trajectory type (chirp, random, or square).",
            "Each fold contains all 50-sample windows from all CSV runs of that trajectory.",
            "y0 is the drone state at the step immediately preceding each window.",
            "Cross-validation is leave-one-trajectory-out over the 3 training trajectory types.",
            "Test trajectory is 'melon', held out from training and validation.",
            "Metric: MAE averaged over all windows, all 50 time steps, and all 12 outputs.",
        ],
    }
    (base_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_trajectory_overview(
    sequence: DroneSequence,
    plots_path: Path,
) -> None:
    plots_path.mkdir(parents=True, exist_ok=True)
    y_flat = sequence.y.reshape(-1, sequence.y.shape[-1]).numpy()
    T = y_flat.shape[0]
    time_axis = np.arange(T)

    n_cols = 3
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(output_names):
        axes[i].plot(time_axis, y_flat[:, i], linewidth=0.8)
        axes[i].set_ylabel(name, fontsize=8)
        axes[i].grid(True, alpha=0.25)

    axes[-1].set_xlabel("time step")
    fig.suptitle(f"{sequence.name} — all windows flattened")
    fig.tight_layout()
    fig.savefig(plots_path / f"{sequence.name}_trajectory.png", dpi=120)
    plt.close(fig)


def main() -> None:
    set_global_seed(config_pars_general["seed"])

    base_path = Path(config_pars_general["base_path"])
    checkpoint_path = Path(config_pars_general["checkpoint_path"])
    plots_path = Path(config_pars_general["plots_path"])
    log_dir = Path(config_pars_general["log_dir"])

    for path in (checkpoint_path, plots_path, log_dir):
        if path.exists():
            shutil.rmtree(path)

    for path in (base_path, checkpoint_path, plots_path, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_sequences_dir = base_path / "train_sequences"
    test_sequences_dir = base_path / "test_sequences"
    if train_sequences_dir.exists():
        shutil.rmtree(train_sequences_dir)
    if test_sequences_dir.exists():
        shutil.rmtree(test_sequences_dir)

    seq_len = int(config_pars_general["seq_len"])

    print("Loading training trajectories...")
    train_sequences: dict[str, DroneSequence] = {}
    for name in config_pars_general["train_list"]:
        print(f"  {name}")
        seq = load_trajectory(TRAIN_DATA_DIR, name, seq_len)
        train_sequences[name] = seq
        print(f"    windows={seq.num_sequences}  u={tuple(seq.u.shape)}  y={tuple(seq.y.shape)}")

    print("\nLoading test trajectories...")
    test_sequences: dict[str, DroneSequence] = {}
    for name in config_pars_general["test_list"]:
        print(f"  {name}")
        seq = load_trajectory(TEST_DATA_DIR, name, seq_len)
        test_sequences[name] = seq
        print(f"    windows={seq.num_sequences}  u={tuple(seq.u.shape)}  y={tuple(seq.y.shape)}")

    for name, seq in train_sequences.items():
        save_sequence(train_sequences_dir / f"{name}.npz", seq)
    for name, seq in test_sequences.items():
        save_sequence(test_sequences_dir / f"{name}.npz", seq)

    for seq in list(train_sequences.values()) + list(test_sequences.values()):
        plot_trajectory_overview(seq, plots_path)

    save_config(config_pars_general)
    save_metadata(config_pars_general, train_sequences, test_sequences)

    print(f"\nPrepared {config_pars_general['benchmark_name']}")
    print(f"Saved cached datasets in {base_path}")
    print(f"Saved plots in {plots_path}")


if __name__ == "__main__":
    main()
