from __future__ import annotations

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
import torch

try:
    import nonlinear_benchmarks
except ImportError as exc:
    raise SystemExit(
        "Missing dependency `nonlinear_benchmarks`. Install the benchmark requirements first."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "data"
DOWNLOAD_DIR = RAW_DATA_DIR / "downloads"

input_names = ["pump_voltage"]
output_names = ["tank2_level"]

config_pars_general = {
    "benchmark_name": "Cascaded_Tanks",
    "device": "cpu",
    "seed": 42,
    "num_folds": 2,
    "history_window": 5,
    "benchmark_test_initialization_window_length": 50,
    "warmup_test": 45, # given by  benchmark_test_initialization_window_length - history_window
    "validation_metric": "rmse",
    "base_path": "./cached_data",
    "raw_data_path": "./data",
    "checkpoint_path": "./checkpoints",
    "plots_path": "./plots",
    "log_dir": "./logs",
    "eval_every": 25,
    "fold_time_budget_seconds": 800.0,
    "n_inputs": 1,
    "n_outputs": 1,
    "n_states": 10, # number of past input + output used to create initial conditions 
}


@dataclass
class TankSequence:
    name: str
    u: torch.Tensor
    y: torch.Tensor
    y0: torch.Tensor
    sampling_time: float
    warmup: int = 0
    start_index: int = 0
    stop_index: int = 0

    @property
    def num_samples(self) -> int:
        return int(self.u.shape[1])

    def clone(self) -> "TankSequence":
        return TankSequence(
            name=self.name,
            u=self.u.clone(),
            y=self.y.clone(),
            y0=self.y0.clone(),
            sampling_time=self.sampling_time,
            warmup=self.warmup,
            start_index=self.start_index,
            stop_index=self.stop_index,
        )


class Normalizer:
    def __init__(
        self,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        history_window: int,
        n_inputs: int,
        n_outputs: int,
        eps: float = 1e-6,
    ) -> None:
        self.u_mean = u_mean.float()
        self.u_std = torch.clamp(u_std.float(), min=eps)
        self.y_mean = y_mean.float()
        self.y_std = torch.clamp(y_std.float(), min=eps)
        self.history_window = int(history_window)
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.eps = eps

    @classmethod
    def fit(cls, sequences: list[TankSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")

        u_all = torch.cat([sequence.u.reshape(-1, sequence.u.shape[-1]) for sequence in sequences], dim=0)
        y_all = torch.cat([sequence.y.reshape(-1, sequence.y.shape[-1]) for sequence in sequences], dim=0)

        u_mean = u_all.mean(dim=0, keepdim=True)
        u_std = u_all.std(dim=0, keepdim=True, unbiased=False)
        y_mean = y_all.mean(dim=0, keepdim=True)
        y_std = y_all.std(dim=0, keepdim=True, unbiased=False)

        return cls(
            u_mean=u_mean,
            u_std=u_std,
            y_mean=y_mean,
            y_std=y_std,
            history_window=history_window,
            n_inputs=int(sequences[0].u.shape[-1]),
            n_outputs=int(sequences[0].y.shape[-1]),
            eps=eps,
        )

    def normalize_sequence(self, sequence: TankSequence) -> TankSequence:
        normalized = sequence.clone()
        normalized.u = (normalized.u - self.u_mean.unsqueeze(0)) / self.u_std.unsqueeze(0)
        normalized.y = (normalized.y - self.y_mean.unsqueeze(0)) / self.y_std.unsqueeze(0)

        u_hist_dim = self.history_window * self.n_inputs
        y_hist_dim = self.history_window * self.n_outputs

        u_mean_hist = self.u_mean.repeat(1, self.history_window)
        u_std_hist = self.u_std.repeat(1, self.history_window)
        y_mean_hist = self.y_mean.repeat(1, self.history_window)
        y_std_hist = self.y_std.repeat(1, self.history_window)

        ic_mean = torch.cat([u_mean_hist, y_mean_hist], dim=1).unsqueeze(0)
        ic_std = torch.cat([u_std_hist, y_std_hist], dim=1).unsqueeze(0)

        if sequence.y0.shape[-1] != u_hist_dim + y_hist_dim:
            raise ValueError("Initial-condition vector size is inconsistent with history_window.")

        normalized.y0 = (normalized.y0 - ic_mean) / ic_std
        return normalized

    def denormalize_y_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        std = self.y_std.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        return values * std + mean

    def state_dict(self) -> dict[str, np.ndarray | float | int]:
        return {
            "u_mean": self.u_mean.detach().cpu().numpy(),
            "u_std": self.u_std.detach().cpu().numpy(),
            "y_mean": self.y_mean.detach().cpu().numpy(),
            "y_std": self.y_std.detach().cpu().numpy(),
            "history_window": int(self.history_window),
            "n_inputs": int(self.n_inputs),
            "n_outputs": int(self.n_outputs),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray | float | int]) -> "Normalizer":
        return cls(
            u_mean=torch.as_tensor(state["u_mean"], dtype=torch.float32),
            u_std=torch.as_tensor(state["u_std"], dtype=torch.float32),
            y_mean=torch.as_tensor(state["y_mean"], dtype=torch.float32),
            y_std=torch.as_tensor(state["y_std"], dtype=torch.float32),
            history_window=int(state["history_window"]),
            n_inputs=int(state["n_inputs"]),
            n_outputs=int(state["n_outputs"]),
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_official_splits(force_download: bool = False):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    train_tuple, test_tuple = nonlinear_benchmarks.Cascaded_Tanks(
        atleast_2d=True,
        always_return_tuples_of_datasets=True,
        dir_placement=str(DOWNLOAD_DIR),
        force_download=force_download,
    )
    return train_tuple[0], test_tuple[0]


def benchmark_to_arrays(dataset) -> tuple[np.ndarray, np.ndarray, float]:
    u = np.asarray(dataset.u, dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.float32)

    if u.ndim == 1:
        u = u[:, None]
    if y.ndim == 1:
        y = y[:, None]

    return u, y, float(dataset.sampling_time)


def build_initial_condition(
    u_full: np.ndarray,
    y_full: np.ndarray,
    start_index: int,
    history_window: int,
) -> np.ndarray:
    if start_index < history_window:
        raise ValueError("Not enough past samples to build the initial-condition vector.")

    u_history = [u_full[start_index - lag] for lag in range(1, history_window + 1)]
    y_history = [y_full[start_index - lag] for lag in range(1, history_window + 1)]

    ic_vector = np.concatenate(
        [
            np.concatenate(u_history, axis=0),
            np.concatenate(y_history, axis=0),
        ],
        axis=0,
    )
    return ic_vector.astype(np.float32)


def effective_test_warmup(history_window: int, benchmark_window_length: int) -> int:
    """Translate the benchmark warmup to the cropped test sequence used by this repo."""
    return max(0, int(benchmark_window_length) - int(history_window))


def make_sequence(
    name: str,
    u_full: np.ndarray,
    y_full: np.ndarray,
    sampling_time: float,
    start_index: int,
    stop_index: int,
    history_window: int,
    warmup: int = 0,
) -> TankSequence:
    y0_vector = build_initial_condition(
        u_full=u_full,
        y_full=y_full,
        start_index=start_index,
        history_window=history_window,
    )
    return TankSequence(
        name=name,
        u=torch.from_numpy(u_full[start_index:stop_index]).unsqueeze(0),
        y=torch.from_numpy(y_full[start_index:stop_index]).unsqueeze(0),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1),
        sampling_time=sampling_time,
        warmup=warmup,
        start_index=start_index,
        stop_index=stop_index,
    )


def split_train_into_folds(
    u_full: np.ndarray,
    y_full: np.ndarray,
    sampling_time: float,
    num_folds: int,
    history_window: int,
) -> dict[str, TankSequence]:
    usable_start = history_window
    usable_length = len(u_full) - history_window
    fold_sizes = np.full(num_folds, usable_length // num_folds, dtype=int)
    fold_sizes[: usable_length % num_folds] += 1

    train_sequences: dict[str, TankSequence] = {}
    current_start = usable_start
    for fold_index, fold_size in enumerate(fold_sizes, start=1):
        current_stop = current_start + int(fold_size)
        name = f"fold_{fold_index}"
        train_sequences[name] = make_sequence(
            name=name,
            u_full=u_full,
            y_full=y_full,
            sampling_time=sampling_time,
            start_index=current_start,
            stop_index=current_stop,
            history_window=history_window,
            warmup=0,
        )
        current_start = current_stop

    return train_sequences


def save_sequence(path: Path, sequence: TankSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.asarray(sequence.name),
        u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32),
        y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time),
        warmup=np.int64(sequence.warmup),
        start_index=np.int64(sequence.start_index),
        stop_index=np.int64(sequence.stop_index),
    )


def load_sequence(path: Path) -> TankSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")

    data = np.load(path, allow_pickle=True)
    name_value = data["name"]
    name = name_value.item() if np.asarray(name_value).shape == () else str(name_value)

    return TankSequence(
        name=str(name),
        u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
        y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)),
        y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
        sampling_time=float(data["sampling_time"]),
        warmup=int(data["warmup"]),
        start_index=int(data.get("start_index", 0)),
        stop_index=int(data.get("stop_index", 0)),
    )


def save_raw_arrays(path: Path, u: np.ndarray, y: np.ndarray, sampling_time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=u.astype(np.float32),
        y=y.astype(np.float32),
        sampling_time=np.float32(sampling_time),
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


def load_train_sequences(base_path: str | Path | None = None) -> dict[str, TankSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached training sequences under {train_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_datasets_and_config(
    base_path: str | Path | None = None,
) -> tuple[dict[str, TankSequence], TankSequence, dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    test_sequence = load_sequence(root / "dataset_test.npz")
    config = load_config(root)

    reference_train = next(iter(train_sequences.values()))
    config["n_inputs"] = int(reference_train.u.shape[-1])
    config["n_outputs"] = int(reference_train.y.shape[-1])
    config["n_states"] = int(reference_train.y0.shape[-1])
    for sequence in train_sequences.values():
        sequence.warmup = 0
    benchmark_window_length = int(
        config.get(
            "benchmark_test_initialization_window_length",
            config_pars_general["benchmark_test_initialization_window_length"],
        )
    )
    config["benchmark_test_initialization_window_length"] = benchmark_window_length
    config["warmup_test"] = effective_test_warmup(
        history_window=int(config["history_window"]),
        benchmark_window_length=benchmark_window_length,
    )
    test_sequence.warmup = int(config["warmup_test"])

    return train_sequences, test_sequence, config


def load_train_sequences_and_config(
    base_path: str | Path | None = None,
) -> tuple[dict[str, TankSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    config = load_config(root)

    reference_train = next(iter(train_sequences.values()))
    config["n_inputs"] = int(reference_train.u.shape[-1])
    config["n_outputs"] = int(reference_train.y.shape[-1])
    config["n_states"] = int(reference_train.y0.shape[-1])
    for sequence in train_sequences.values():
        sequence.warmup = 0
    benchmark_window_length = int(
        config.get(
            "benchmark_test_initialization_window_length",
            config_pars_general["benchmark_test_initialization_window_length"],
        )
    )
    config["benchmark_test_initialization_window_length"] = benchmark_window_length
    config["warmup_test"] = effective_test_warmup(
        history_window=int(config["history_window"]),
        benchmark_window_length=benchmark_window_length,
    )

    return train_sequences, config


def plot_full_trajectory(
    name: str,
    u: np.ndarray,
    y: np.ndarray,
    sampling_time: float,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(len(u), dtype=np.float32) * sampling_time

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_axis, u[:, 0], linewidth=1.5, color="#006d77")
    axes[0].set_ylabel(input_names[0])
    axes[0].set_title(f"{name} input trajectory")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y[:, 0], linewidth=1.5, color="#bc6c25")
    axes[1].set_ylabel(output_names[0])
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"{name} output trajectory")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_metadata(
    config: dict,
    train_sequences: dict[str, TankSequence],
    test_sequence: TankSequence,
    full_train_length: int,
) -> None:
    base_path = Path(config["base_path"])
    metadata = {
        "benchmark_name": config["benchmark_name"],
        "validation_metric": config["validation_metric"],
        "device": config["device"],
        "num_folds": config["num_folds"],
        "history_window": config["history_window"],
        "sampling_time": test_sequence.sampling_time,
        "full_train_length": int(full_train_length),
        "test_sequence_length": int(test_sequence.num_samples),
        "initial_condition_dimension": int(test_sequence.y0.shape[-1]),
        "input_names": input_names,
        "output_names": output_names,
        "train_sequences": [
            {
                "name": sequence.name,
                "start_index": int(sequence.start_index),
                "stop_index": int(sequence.stop_index),
                "num_samples": int(sequence.num_samples),
            }
            for sequence in train_sequences.values()
        ],
        "notes": [
            "The full training trajectory is split into 2 contiguous folds.",
            "Each fold starts only after 5 past samples are available.",
            "The initial-condition vector is [u(k-1)...u(k-5), y(k-1)...y(k-5)].",
            "Cross-validation is standard leave-one-sequence-out over the 2 folds.",
        ],
    }
    (base_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    set_global_seed(config_pars_general["seed"])

    base_path = Path(config_pars_general["base_path"])
    raw_data_path = Path(config_pars_general["raw_data_path"])
    plots_path = Path(config_pars_general["plots_path"])
    checkpoint_path = Path(config_pars_general["checkpoint_path"])
    log_dir = Path(config_pars_general["log_dir"])

    for path in (plots_path, checkpoint_path, log_dir):
        if path.exists():
            shutil.rmtree(path)

    for path in (base_path, raw_data_path, plots_path, checkpoint_path, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_sequences_dir = base_path / "train_sequences"
    if train_sequences_dir.exists():
        shutil.rmtree(train_sequences_dir)

    train_raw, test_raw = load_official_splits(force_download=False)
    u_train_full, y_train_full, sampling_time_train = benchmark_to_arrays(train_raw.atleast_2d())
    u_test_full, y_test_full, sampling_time_test = benchmark_to_arrays(test_raw.atleast_2d())
    print(
        "Official test state_initialization_window_length="
        f"{getattr(test_raw, 'state_initialization_window_length', 'MISSING')}"
    )

    history_window = int(config_pars_general["history_window"])
    benchmark_window_length = int(getattr(test_raw, "state_initialization_window_length", 0))
    config_pars_general["benchmark_test_initialization_window_length"] = benchmark_window_length
    config_pars_general["warmup_test"] = effective_test_warmup(
        history_window=history_window,
        benchmark_window_length=benchmark_window_length,
    )
    train_sequences = split_train_into_folds(
        u_full=u_train_full,
        y_full=y_train_full,
        sampling_time=sampling_time_train,
        num_folds=config_pars_general["num_folds"],
        history_window=history_window,
    )
    test_sequence = make_sequence(
        name="test",
        u_full=u_test_full,
        y_full=y_test_full,
        sampling_time=sampling_time_test,
        start_index=history_window,
        stop_index=len(u_test_full),
        history_window=history_window,
        warmup=config_pars_general["warmup_test"],
    )
    print(f"Loaded test dataset with warmup_test={test_sequence.warmup}")

    for name, sequence in train_sequences.items():
        save_sequence(train_sequences_dir / f"{name}.npz", sequence)
    save_sequence(base_path / "dataset_test.npz", test_sequence)

    save_raw_arrays(raw_data_path / "train_full_raw.npz", u_train_full, y_train_full, sampling_time_train)
    save_raw_arrays(raw_data_path / "test_full_raw.npz", u_test_full, y_test_full, sampling_time_test)

    plot_full_trajectory("train_full", u_train_full, y_train_full, sampling_time_train, plots_path / "train_trajectory.png")
    plot_full_trajectory("test_full", u_test_full, y_test_full, sampling_time_test, plots_path / "test_trajectory.png")

    config_pars_general["n_states"] = history_window * (
        config_pars_general["n_inputs"] + config_pars_general["n_outputs"]
    )
    save_config(config_pars_general)
    save_metadata(config_pars_general, train_sequences, test_sequence, full_train_length=len(u_train_full))

    print(f"Prepared {config_pars_general['benchmark_name']}")
    for name, sequence in train_sequences.items():
        print(
            f"{name}: start={sequence.start_index} stop={sequence.stop_index} "
            f"shape_u={tuple(sequence.u.shape)} shape_y={tuple(sequence.y.shape)} shape_ic={tuple(sequence.y0.shape)}"
        )
    print(
        f"test: start={test_sequence.start_index} stop={test_sequence.stop_index} "
        f"shape_u={tuple(test_sequence.u.shape)} shape_y={tuple(test_sequence.y.shape)} shape_ic={tuple(test_sequence.y0.shape)}"
    )
    print(f"Saved cached datasets in {base_path}")
    print(f"Saved trajectory plots in {plots_path}")


if __name__ == "__main__":
    main()
