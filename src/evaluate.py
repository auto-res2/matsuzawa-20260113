"""src/evaluate.py
Independent evaluation script for Hydra-based experiments.
- Executes via: uv run python -m src.evaluate results_dir={path} run_ids='["run-1", ...]'
- Fetches data from WandB (or fallback to local metrics.jsons) and writes per-run metrics and aggregated metrics.
"""
import argparse
import json
import os
from pathlib import Path
import io
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml

from typing import List


def _load_root_config() -> dict:
    # Load repository root config.yaml if present
    root = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if not root.exists():
        return {}
    with open(root, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
    return data


def _export_history_json(history_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = history_df.to_dict(orient="records")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)


def _export_summary_json(summary: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def _plot_learning_curve(history_df: pd.DataFrame, out_path: Path, key_x: str = "epoch", keys: List[str] = None) -> None:
    if history_df is None or history_df.empty:
        return
    if keys is None:
        keys = ["train_acc", "val_acc"]
    plt.figure()
    for k in keys:
        if k in history_df.columns:
            plt.plot(history_df[key_x], history_df[k], label=k)
    plt.xlabel(key_x)
    plt.ylabel("metric")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Directory containing outputs from train runs")
    parser.add_argument("--run_ids", required=True, help="JSON string list of run_ids to evaluate")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_ids = json.loads(args.run_ids)

    root_cfg = _load_root_config()
    wandb_entity = root_cfg.get("wandb", {}).get("entity", None)
    wandb_project = root_cfg.get("wandb", {}).get("project", None)

    # STEP 1: Per-run processing: collect per-run metrics and figures
    aggregated = {
        "primary_metric": root_cfg.get("primary_metric", "accuracy"),
        "metrics": {},
        "best_proposed": {"run_id": None, "value": None},
        "best_baseline": {"run_id": None, "value": None},
        "gap": None,
    }

    per_run = {}
    for run_id in run_ids:
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = run_dir / "metrics.json"
        if metrics_json.exists():
            with open(metrics_json, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}
        acc = None
        # try common keys
        for key in ["val_acc", "val_accuracy", "accuracy"]:
            if key in metrics:
                acc = metrics[key]
                break
        per_run[run_id] = {"acc": acc, "path": str(run_dir)}
        aggregated["metrics"].setdefault("accuracy", {})[run_id] = acc

    # Identify best proposed vs best baseline
    best_proposed = None
    best_proposed_val = -1.0
    best_baseline = None
    best_baseline_val = -1.0
    for run_id, data in per_run.items():
        val = data["acc"]
        if val is None:
            continue
        if "proposed" in run_id:
            if val > best_proposed_val:
                best_proposed_val = val
                best_proposed = run_id
        if "baseline" in run_id or "comparative" in run_id or "comparative" in run_id:
            if val > best_baseline_val:
                best_baseline_val = val
                best_baseline = run_id

    aggregated["best_proposed"] = {"run_id": best_proposed, "value": best_proposed_val if best_proposed is not None else None}
    aggregated["best_baseline"] = {"run_id": best_baseline, "value": best_baseline_val if best_baseline is not None else None}

    if best_proposed is not None and best_baseline is not None and best_baseline_val != 0:
        aggregated["gap"] = ((best_proposed_val - best_baseline_val) / best_baseline_val) * 100

    out_agg = results_dir / "comparison" / "aggregated_metrics.json"
    out_agg.parent.mkdir(parents=True, exist_ok=True)
    with open(out_agg, "w") as f:
        json.dump(aggregated, f, indent=2)

    # Step 2: per-run figures (simple learning curves placeholder)
    for run_id in run_ids:
        run_dir = results_dir / run_id
        hist_path = run_dir / "history.json"
        if hist_path.exists():
            hist_df = pd.read_json(hist_path)
            fig_path = results_dir / "comparison" / f"{run_id}_learning_curve.pdf"
            _plot_learning_curve(hist_df, fig_path, key_x="epoch")

    print("Evaluation complete. Outputs written under:", results_dir)


if __name__ == "__main__":
    main()
