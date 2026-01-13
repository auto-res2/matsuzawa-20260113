"""src/main.py
Hydra-driven experiment launcher. It reads a run_id via CLI/Hydra and launches train.py as a subprocess
with a final merged YAML config.

Usage (from repo root):
  uv run python -u -m src.main run={run_id} results_dir={path} mode=full
"""
import subprocess
import sys
import time
import json
import os
from pathlib import Path
import yaml
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> int:
    # Allow dynamic attributes (disable struct mode)
    OmegaConf.set_struct(cfg, False)
    # Run metadata - support both 'run' and 'run_id' parameters
    run_id = None
    if getattr(cfg, "run", None) is not None:
        run_id = str(cfg.run)
    elif getattr(cfg, "run_id", None) is not None:
        run_id = str(cfg.run_id)

    if run_id is None:
        raise ValueError("run_id must be provided via cfg.run or cfg.run_id (e.g., uv run ... run=run-1)")

    results_dir = Path(str(cfg.results_dir)) if getattr(cfg, "results_dir", None) is not None else Path("./results")
    mode = str(cfg.mode) if getattr(cfg, "mode", None) is not None else "full"

    runs_dir = Path("config/runs")
    run_yaml_path = runs_dir / f"{run_id}.yaml"
    run_overrides = {}
    if run_yaml_path.exists():
        with open(run_yaml_path, 'r') as f:
            run_overrides = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Run YAML not found for {run_id} at {run_yaml_path}. Proceeding with base config.")

    # Merge base cfg with run-specific overrides into a final YAML for training
    base_container = OmegaConf.to_container(cfg, resolve=True)
    final_cfg = OmegaConf.merge(OmegaConf.create(base_container), OmegaConf.create(run_overrides))

    results_dir.mkdir(parents=True, exist_ok=True)
    final_cfg_path = results_dir / f"{run_id}.final.yaml"
    with open(final_cfg_path, 'w') as f:
        f.write(OmegaConf.to_yaml(final_cfg))

    train_cmd = [sys.executable, "-u", "-m", "src.train",
                 "--final_cfg", str(final_cfg_path),
                 "--run_id", run_id,
                 "--results_dir", str(results_dir),
                 "--mode", mode]

    print(f"Launching training for run {run_id} mode={mode}")
    print("Command:", " ".join(train_cmd))

    start = time.time()
    proc = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    elapsed = time.time() - start
    print(f"Training finished for run {run_id} in {elapsed:.2f}s (return code {proc.returncode})")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
