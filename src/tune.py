from __future__ import annotations

"""
tune.py

Grid-search validation tuning for the AMI diarization pipeline.

This version reuses runner.py functions instead of duplicating the pipeline.

Example:
    python ./src/tune.py --model-type ecapa --recordings ES2002a ES2002b ES2002c

For WavLM:
    python ./src/tune.py --model-type wavlm --recordings ES2002a ES2002b ES2002c
"""

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import torch

from runner import build_dataset, build_model, run_single_recording
from debug import set_verbose, vprint


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TuneConfig:
    window_sec: float
    hop_sec: float
    smoothing_kernel: int
    model_type: str
    clustering: str = "kmeans"
    n_neighbors: int = 10
    merge_gap: float = 0.25
    min_seg_dur: float = 0.75


def make_grid(
    model_type: str,
    clustering_values: Iterable[str],
    window_values: Iterable[float],
    hop_values: Iterable[float],
    smoothing_values: Iterable[int],
    n_neighbors_values: Iterable[int],
    merge_gap_values: Iterable[float],
    min_seg_dur_values: Iterable[float],
) -> List[TuneConfig]:
    configs: List[TuneConfig] = []

    for clustering in clustering_values:
        for window_sec in window_values:
            for hop_sec in hop_values:
                if hop_sec > window_sec:
                    continue

                for smoothing_kernel in smoothing_values:
                    for n_neighbors in n_neighbors_values:
                        for merge_gap in merge_gap_values:
                            for min_seg_dur in min_seg_dur_values:
                                configs.append(
                                    TuneConfig(
                                        window_sec=window_sec,
                                        hop_sec=hop_sec,
                                        smoothing_kernel=smoothing_kernel,
                                        model_type=model_type,
                                        clustering=clustering,
                                        n_neighbors=n_neighbors,
                                        merge_gap=merge_gap,
                                        min_seg_dur=min_seg_dur,
                                    )
                                )

    return configs


def write_rows_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(values, default: List[float]) -> List[float]:
    if not values:
        return default
    return [float(v) for v in values]


def parse_int_list(values, default: List[int]) -> List[int]:
    if not values:
        return default
    return [int(v) for v in values]


def evaluate_config_on_recording(
    audio_dir: str,
    annotation_dir: str,
    recording_id: str,
    config: TuneConfig,
    device: str,
    vad_threshold: float,
    cache_dir: Path,
    collar: float,
    ignore_overlap: bool,
):
    dataset = build_dataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        recording_id=recording_id,
        target_sr=16000,
    )

    model = build_model(
        model_type=config.model_type,
        device=device,
        window_sec=config.window_sec,
        hop_sec=config.hop_sec,
        vad_threshold=vad_threshold,
        smoothing_kernel=config.smoothing_kernel,
        cache_dir=str(cache_dir),
        use_embedding_cache=True,
        clustering_method=config.clustering,
        n_neighbors=config.n_neighbors,
        merge_gap=config.merge_gap,
        min_seg_dur=config.min_seg_dur,
    )

    sample, result, metrics, mapping = run_single_recording(
        dataset=dataset,
        recording_id=recording_id,
        model=model,
        collar=collar,
        ignore_overlap=ignore_overlap,
    )

    return {
        "recording_id": recording_id,
        "DER": float(metrics["DER"]),
        "miss": float(metrics["miss"]),
        "false_alarm": float(metrics["false_alarm"]),
        "confusion": float(metrics["confusion"]),
        "ref_total": float(metrics["ref_total"]),
        "scored_frames": float(metrics["scored_frames"]),
        "num_pred_segments": float(len(result.segments)),
        "cluster_mapping": str(mapping),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search validation tuning for AMI diarization."
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default=f"{ROOT}/data/amicorpus",
        help="Path to AMI corpus root or recording audio folder.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=f"{ROOT}/data/ami_public_manual_1.6.2",
        help="Path to AMI annotation directory.",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["ES2002a"],
        help="Validation recording IDs, e.g. ES2002a ES2002b ES2002c.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ecapa",
        choices=["baseline", "ecapa", "wavlm", "advanced"],
        help="Architecture/backend to tune.",
    )

    parser.add_argument(
        "--clustering-values",
        nargs="+",
        default=["kmeans"],
        choices=["kmeans", "spectral"],
    )
    parser.add_argument("--window-values", nargs="+", default=None)
    parser.add_argument("--hop-values", nargs="+", default=None)
    parser.add_argument("--smoothing-values", nargs="+", default=None)
    parser.add_argument("--n-neighbors-values", nargs="+", default=None)
    parser.add_argument("--merge-gap-values", nargs="+", default=None)
    parser.add_argument("--min-seg-dur-values", nargs="+", default=None)

    parser.add_argument("--vad-threshold", type=float, default=8e-5)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        help="Score overlap regions too. Default ignores overlap.",
    )
    parser.add_argument("--cache-dir", type=str, default=f"{ROOT}/outputs/cache")
    parser.add_argument("--out-csv", type=str, default=f"{ROOT}/outputs/tuning_results.csv")
    parser.add_argument("--summary-csv", type=str, default=f"{ROOT}/outputs/tuning_summary.csv")
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    set_verbose(args.verbose)

    window_values = parse_float_list(args.window_values, [1.5, 2.0, 3.0])
    hop_values = parse_float_list(args.hop_values, [0.75, 1.0, 1.5])
    smoothing_values = parse_int_list(args.smoothing_values, [1, 3, 5])
    n_neighbors_values = parse_int_list(args.n_neighbors_values, [10])
    merge_gap_values = parse_float_list(args.merge_gap_values, [0.25])
    min_seg_dur_values = parse_float_list(args.min_seg_dur_values, [0.75])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ignore_overlap = not args.include_overlap

    configs = make_grid(
        model_type=args.model_type,
        clustering_values=args.clustering_values,
        window_values=window_values,
        hop_values=hop_values,
        smoothing_values=smoothing_values,
        n_neighbors_values=n_neighbors_values,
        merge_gap_values=merge_gap_values,
        min_seg_dur_values=min_seg_dur_values,
    )

    vprint("\n=== Tuning Setup ===")
    vprint(f"recordings:     {args.recordings}")
    vprint(f"model_type:     {args.model_type}")
    vprint(f"device:         {device}")
    vprint(f"num_configs:    {len(configs)}")
    vprint(f"out_csv:        {args.out_csv}")
    vprint(f"summary_csv:    {args.summary_csv}")

    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for config_idx, config in enumerate(configs, start=1):
        config_dict = asdict(config)
        vprint(f"\n=== Config {config_idx}/{len(configs)} ===")
        vprint(str(config_dict))

        config_ders: List[float] = []

        for recording_id in args.recordings:
            vprint(f"\n[Run] recording={recording_id}")

            try:
                metrics = evaluate_config_on_recording(
                    audio_dir=args.audio_dir,
                    annotation_dir=args.annotation_dir,
                    recording_id=recording_id,
                    config=config,
                    device=device,
                    vad_threshold=args.vad_threshold,
                    cache_dir=Path(args.cache_dir),
                    collar=args.collar,
                    ignore_overlap=ignore_overlap,
                )

                config_ders.append(float(metrics["DER"]))

                row = {
                    "config_id": config_idx,
                    **config_dict,
                    **metrics,
                    "error": "",
                }
                all_rows.append(row)

                vprint(
                    f"[Result] {recording_id} | "
                    f"DER={metrics['DER']:.4f} | "
                    f"miss={metrics['miss']:.0f} | "
                    f"FA={metrics['false_alarm']:.0f} | "
                    f"conf={metrics['confusion']:.0f}"
                )

            except Exception as exc:
                row = {
                    "config_id": config_idx,
                    **config_dict,
                    "recording_id": recording_id,
                    "DER": "",
                    "miss": "",
                    "false_alarm": "",
                    "confusion": "",
                    "ref_total": "",
                    "scored_frames": "",
                    "num_pred_segments": "",
                    "cluster_mapping": "",
                    "error": repr(exc),
                }
                all_rows.append(row)
                vprint(f"[ERROR] {recording_id}: {exc}")

        if config_ders:
            summary = {
                "config_id": config_idx,
                **config_dict,
                "mean_DER": mean(config_ders),
                "std_DER": pstdev(config_ders) if len(config_ders) > 1 else 0.0,
                "num_successful_recordings": len(config_ders),
            }
        else:
            summary = {
                "config_id": config_idx,
                **config_dict,
                "mean_DER": "",
                "std_DER": "",
                "num_successful_recordings": 0,
            }

        summary_rows.append(summary)

        write_rows_csv(Path(args.out_csv), all_rows)
        write_rows_csv(Path(args.summary_csv), summary_rows)

    valid_summaries = [
        row for row in summary_rows
        if isinstance(row["mean_DER"], float)
    ]

    if valid_summaries:
        best = min(valid_summaries, key=lambda r: r["mean_DER"])

        vprint("\n=== Best Validation Config ===")
        for key, value in best.items():
            if isinstance(value, float):
                vprint(f"{key}: {value:.4f}")
            else:
                vprint(f"{key}: {value}")
    else:
        vprint("\nNo successful configs completed.")

    vprint("\nDone.")


if __name__ == "__main__":
    main()
