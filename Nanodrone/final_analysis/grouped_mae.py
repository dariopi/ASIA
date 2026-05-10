"""Stride-1 test MAE grouped by output category for AR-best and RS-best models.

Groups:
  positions         : x, y, z
  angular positions : roll, pitch, yaw
  linear velocities : vx, vy, vz
  angular velocities: wx, wy, wz

Usage:
    python final_analysis/grouped_mae.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import importlib.util
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RS_ROOT   = REPO_ROOT / "random_search"
sys.path.insert(0, str(REPO_ROOT))

import prepare
import test as test_eval

# Load random_search/model.py without polluting sys.path
_rs_spec = importlib.util.spec_from_file_location("rs_model", RS_ROOT / "model.py")
rs_model_module = importlib.util.module_from_spec(_rs_spec)
_rs_spec.loader.exec_module(rs_model_module)

SEQ_LEN  = 50
OUT_DIR  = Path(__file__).resolve().parent

GROUPS = {
    "positions":          ["x", "y", "z"],
    "angular_positions":  ["roll", "pitch", "yaw"],
    "linear_velocities":  ["vx", "vy", "vz"],
    "angular_velocities": ["wx", "wy", "wz"],
}


# ---------------------------------------------------------------------------
# Sliding-window test sequence (stride 1)
# ---------------------------------------------------------------------------

def sliding_windows(data_dir: Path, name: str) -> prepare.DroneSequence:
    import pandas as pd
    u_list, y_list, y0_list = [], [], []
    for fname in sorted(data_dir.iterdir()):
        if fname.name.startswith(name) and fname.suffix == ".csv":
            df = pd.read_csv(fname)
            u_vals = df[prepare.input_names].values.astype(np.float32)
            y_vals = df[prepare.output_names].values.astype(np.float32)
            n = max(0, len(df) - SEQ_LEN)
            u_w  = np.empty((n, SEQ_LEN, len(prepare.input_names)),  dtype=np.float32)
            y_w  = np.empty((n, SEQ_LEN, len(prepare.output_names)), dtype=np.float32)
            y0_w = np.empty((n, 1,       len(prepare.output_names)), dtype=np.float32)
            for i in range(n):
                y0_w[i, 0] = y_vals[i]
                u_w[i]     = u_vals[i + 1: i + 1 + SEQ_LEN]
                y_w[i]     = y_vals[i + 1: i + 1 + SEQ_LEN]
            u_list.append(u_w); y_list.append(y_w); y0_list.append(y0_w)
    return prepare.DroneSequence(
        name=name,
        u=torch.from_numpy(np.concatenate(u_list,  axis=0)),
        y=torch.from_numpy(np.concatenate(y_list,  axis=0)),
        y0=torch.from_numpy(np.concatenate(y0_list, axis=0)),
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_ar(checkpoint_path: Path, seq: prepare.DroneSequence,
               general_config: dict) -> np.ndarray:
    """Autoresearch model — uses root model.py builder."""
    return test_eval.predict_with_checkpoint(checkpoint_path, seq, general_config)


def predict_rs(checkpoint_path: Path, seq: prepare.DroneSequence,
               general_config: dict) -> np.ndarray:
    """Random-search model — uses random_search/model.py builder."""
    ck = torch.load(checkpoint_path, map_location=general_config["device"],
                    weights_only=False)
    model = rs_model_module.build_model_from_config(
        config_pars=ck["model_config"],
        n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"],
        n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    normalizer = prepare.Normalizer.from_state_dict(ck["normalizer"])
    seq_norm = normalizer.normalize_sequence(seq)
    with torch.no_grad():
        y_hat_norm, _ = model(
            seq_norm.u.to(general_config["device"]),
            seq_norm.y0.to(general_config["device"]),
        )
        y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()
    return y_hat


# ---------------------------------------------------------------------------
# Grouped MAE
# ---------------------------------------------------------------------------

def mae_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae_vec = np.mean(np.abs(y_true - y_pred), axis=(0, 1))
    return {name: float(mae_vec[i]) for i, name in enumerate(prepare.output_names)}


def group_mae(per_output: dict[str, float]) -> dict[str, float]:
    return {
        group: float(np.mean([per_output[o] for o in outputs]))
        for group, outputs in GROUPS.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _, _, general_config = prepare.load_datasets_and_config()
    seq = sliding_windows(prepare.TEST_DATA_DIR, "melon")
    print(f"Windows: {seq.u.shape[0]}  (stride=1, seq_len={SEQ_LEN})")

    # ── Autoresearch best ──────────────────────────────────────────────────
    ar_ckpts = test_eval.load_checkpoint_paths(
        Path(general_config["checkpoint_path"]), "best_so_far"
    )
    ar_preds = [predict_ar(cp, seq, general_config) for cp in ar_ckpts]
    ar_ensemble = np.mean(np.stack(ar_preds, axis=0), axis=0)
    y_true = seq.y.numpy()
    ar_per_out = mae_per_output(y_true, ar_ensemble)
    ar_groups  = group_mae(ar_per_out)

    # ── Random search best ─────────────────────────────────────────────────
    rs_best_dir = RS_ROOT / "checkpoints" / "config_0026"
    rs_ckpts = sorted(rs_best_dir.glob("*/model.pt"))
    rs_preds = [predict_rs(cp, seq, general_config) for cp in rs_ckpts]
    rs_ensemble = np.mean(np.stack(rs_preds, axis=0), axis=0)
    rs_per_out = mae_per_output(y_true, rs_ensemble)
    rs_groups  = group_mae(rs_per_out)

    # ── Print ──────────────────────────────────────────────────────────────
    group_labels = {
        "positions":          "Positions (x,y,z)",
        "angular_positions":  "Angular positions (roll,pitch,yaw)",
        "linear_velocities":  "Linear velocities (vx,vy,vz)",
        "angular_velocities": "Angular velocities (wx,wy,wz)",
    }
    print(f"\n{'Group':<38}  {'AR best':>10}  {'RS best':>10}")
    print("-" * 62)
    for key, label in group_labels.items():
        print(f"{label:<38}  {ar_groups[key]:10.6f}  {rs_groups[key]:10.6f}")
    print("-" * 62)
    overall_ar = float(np.mean(list(ar_per_out.values())))
    overall_rs = float(np.mean(list(rs_per_out.values())))
    print(f"{'Overall mean MAE':<38}  {overall_ar:10.6f}  {overall_rs:10.6f}")

    print(f"\n{'Output':<8}  {'AR best':>10}  {'RS best':>10}  group")
    print("-" * 50)
    out_to_group = {o: g for g, outs in GROUPS.items() for o in outs}
    for name in prepare.output_names:
        print(f"{name:<8}  {ar_per_out[name]:10.6f}  {rs_per_out[name]:10.6f}  "
              f"{out_to_group.get(name, '?')}")

    # ── Save ───────────────────────────────────────────────────────────────
    result = {
        "seq_len": SEQ_LEN,
        "stride": 1,
        "num_windows": int(seq.u.shape[0]),
        "autoresearch_best": {
            "num_fold_models": len(ar_ckpts),
            "overall_mae": overall_ar,
            "per_group": ar_groups,
            "per_output": ar_per_out,
        },
        "random_search_best": {
            "config_id": 26,
            "model_class": "PhysicsResidualLSTM",
            "num_fold_models": len(rs_ckpts),
            "overall_mae": overall_rs,
            "per_group": rs_groups,
            "per_output": rs_per_out,
        },
    }
    out_path = OUT_DIR / "grouped_mae.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
