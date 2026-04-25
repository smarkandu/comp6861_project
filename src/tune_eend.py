from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
import yaml

from datasets.ami import AMIDataset
from datasets.eend_dataset import create_eend_dataloaders
from train_eend import train_eend
from debug import set_verbose, vprint


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EENDTuneConfig:
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    patience: int
    threshold: float


def load_yaml(path: str) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_from_grid(grid: dict, key: str, default):
    value = grid.get(key, default)
    return value if isinstance(value, list) else [value]


def make_grid(grid: dict) -> List[EENDTuneConfig]:
    configs = []

    for hidden_dim in list_from_grid(grid, "hidden_dim_values", [128]):
        for num_layers in list_from_grid(grid, "num_layers_values", [2]):
            for num_heads in list_from_grid(grid, "num_heads_values", [4]):
                if int(hidden_dim) % int(num_heads) != 0:
                    continue

                for dropout in list_from_grid(grid, "dropout_values", [0.1]):
                    for lr in list_from_grid(grid, "lr_values", [3e-4]):
                        for batch_size in list_from_grid(grid, "batch_size_values", [1]):
                            for epochs in list_from_grid(grid, "epoch_values", [20]):
                                for patience in list_from_grid(grid, "patience_values", [5]):
                                    for threshold in list_from_grid(grid, "threshold_values", [0.5]):
                                        configs.append(
                                            EENDTuneConfig(
                                                hidden_dim=int(hidden_dim),
                                                num_layers=int(num_layers),
                                                num_heads=int(num_heads),
                                                dropout=float(dropout),
                                                lr=float(lr),
                                                batch_size=int(batch_size),
                                                epochs=int(epochs),
                                                patience=int(patience),
                                                threshold=float(threshold),
                                            )
                                        )

    return configs


def write_rows_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Config-driven TinyEEND tuning.")
    parser.add_argument("--config", type=str, default="configs/tune_eend.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    data = cfg["data"]
    features = cfg["features"]
    runtime = cfg["runtime"]
    grid = cfg["grid"]

    set_verbose(int(runtime.get("verbose", 1)))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    out_csv = ROOT / runtime.get("out_csv", "outputs/eend_tuning_results.csv")
    summary_csv = ROOT / runtime.get("summary_csv", "outputs/eend_tuning_summary.csv")
    model_dir = ROOT / runtime.get("model_dir", "outputs/models/eend_tuning")
    model_dir.mkdir(parents=True, exist_ok=True)

    configs = make_grid(grid)

    vprint("\n=== EEND Tuning Setup ===")
    vprint(f"device:          {device}")
    vprint(f"train_recordings:{data['train_recordings']}")
    vprint(f"val_recordings:  {data['val_recordings']}")
    vprint(f"test_recordings: {data.get('test_recordings', [])}")
    vprint(f"num_configs:     {len(configs)}")
    vprint(f"out_csv:         {out_csv}")
    vprint(f"summary_csv:     {summary_csv}")
    vprint(f"model_dir:       {model_dir}")

    ami = AMIDataset(
        audio_dir=data["audio_dir"],
        annotation_dir=data["annotation_dir"],
        target_sr=int(features.get("sample_rate", 16000)),
    )

    all_rows = []
    summary_rows = []

    for config_idx, config in enumerate(configs, start=1):
        config_dict = asdict(config)

        vprint(f"\n=== EEND Config {config_idx}/{len(configs)} ===")
        vprint(str(config_dict))

        save_path = model_dir / f"eend_config_{config_idx:03d}.pt"

        try:
            train_loader, val_loader, test_loader = create_eend_dataloaders(
                ami_dataset=ami,
                train_recordings=data["train_recordings"],
                val_recordings=data["val_recordings"],
                test_recordings=data.get("test_recordings", []),
                sample_rate=int(features.get("sample_rate", 16000)),
                n_mels=int(features.get("n_mels", 80)),
                hop_length=int(features.get("hop_length", 160)),
                num_speakers=int(features.get("num_speakers", 4)),
                batch_size=config.batch_size,
            )

            model, best_val_loss = train_eend(
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=int(features.get("n_mels", 80)),
                num_speakers=int(features.get("num_speakers", 4)),
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                epochs=config.epochs,
                lr=config.lr,
                device=device,
                save_path=str(save_path),
                patience=config.patience,
            )

            row = {
                "config_id": config_idx,
                **config_dict,
                "best_val_loss": float(best_val_loss),
                "save_path": str(save_path),
                "error": "",
            }

            all_rows.append(row)
            summary_rows.append(row)

            vprint(
                f"[Result] config={config_idx} | "
                f"best_val_loss={best_val_loss:.4f} | "
                f"saved={save_path}"
            )

        except Exception as exc:
            row = {
                "config_id": config_idx,
                **config_dict,
                "best_val_loss": "",
                "save_path": str(save_path),
                "error": repr(exc),
            }

            all_rows.append(row)
            summary_rows.append(row)

            vprint(f"[ERROR] Config {config_idx}: {exc}")

        write_rows_csv(out_csv, all_rows)
        write_rows_csv(summary_csv, summary_rows)

    valid = [
        row for row in summary_rows
        if isinstance(row["best_val_loss"], float)
    ]

    if valid:
        best = min(valid, key=lambda r: r["best_val_loss"])

        vprint("\n=== Best EEND Validation Config ===")
        for key, value in best.items():
            vprint(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    else:
        vprint("\nNo successful EEND configs completed.")

    vprint("\nDone.")


if __name__ == "__main__":
    main()