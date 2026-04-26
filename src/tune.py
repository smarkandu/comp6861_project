from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import torch
import yaml

from runner import build_dataset, build_model, run_single_recording
from models.vad import build_speech_region_selector
from debug import set_verbose, vprint, set_seed


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TuneConfig:
    model_type: str
    speech_source: str
    vad_threshold: float
    min_speech_overlap: float
    window_sec: float
    hop_sec: float
    smoothing_kernel: int
    clustering_method: str
    n_neighbors: int
    merge_gap: float
    min_seg_dur: float
    use_cache: bool
    config_stem: str


def load_yaml(path: str) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_from_grid(grid: dict, key: str, default):
    value = grid.get(key, default)
    return value if isinstance(value, list) else [value]


def make_grid(grid: dict, runtime: dict, config_stem: str) -> List[TuneConfig]:
    configs = []

    for model_type in list_from_grid(grid, "model_types", ["ecapa"]):
        for speech_source in list_from_grid(grid, "speech_sources", ["oracle"]):
            for vad_threshold in list_from_grid(grid, "vad_threshold_values", [8e-5]):
                for min_speech_overlap in list_from_grid(grid, "min_speech_overlap_values", [0.5]):
                    for window_sec in list_from_grid(grid, "window_values", [3.0]):
                        for hop_sec in list_from_grid(grid, "hop_values", [1.5]):
                            if hop_sec > window_sec:
                                continue

                            for smoothing_kernel in list_from_grid(grid, "smoothing_values", [1]):
                                for clustering_method in list_from_grid(grid, "clustering_values", ["spectral"]):
                                    for n_neighbors in list_from_grid(grid, "n_neighbors_values", [10]):
                                        for merge_gap in list_from_grid(grid, "merge_gap_values", [0.25]):
                                            for min_seg_dur in list_from_grid(grid, "min_seg_dur_values", [0.75]):
                                                for use_cache in list_from_grid(runtime, "use_cache_values", [False]):
                                                    configs.append(
                                                        TuneConfig(
                                                            model_type=model_type,
                                                            speech_source=speech_source,
                                                            vad_threshold=float(vad_threshold),
                                                            min_speech_overlap=float(min_speech_overlap),
                                                            window_sec=float(window_sec),
                                                            hop_sec=float(hop_sec),
                                                            smoothing_kernel=int(smoothing_kernel),
                                                            clustering_method=clustering_method,
                                                            n_neighbors=int(n_neighbors),
                                                            merge_gap=float(merge_gap),
                                                            min_seg_dur=float(min_seg_dur),
                                                            use_cache=bool(use_cache),
                                                            config_stem=config_stem
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


def evaluate_config_on_recording(
    audio_dir: str,
    annotation_dir: str,
    recording_id: str,
    config: TuneConfig,
    device: str,
    cache_dir: Path,
    collar: float,
    ignore_overlap: bool,
):
    dataset, _ = build_dataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        recording_ids=recording_id,
        target_sr=16000,
    )

    model = build_model(
        model_type=config.model_type,
        device=device,
        window_sec=config.window_sec,
        hop_sec=config.hop_sec,
        vad_threshold=config.vad_threshold,
        smoothing_kernel=config.smoothing_kernel,
        cache_dir=str(cache_dir / config.speech_source),
        use_embedding_cache=config.use_cache,
        clustering_method=config.clustering_method,
        n_neighbors=config.n_neighbors,
        merge_gap=config.merge_gap,
        min_seg_dur=config.min_seg_dur,
    )

    speech_selector = build_speech_region_selector(
        source=config.speech_source,
        vad_threshold=config.vad_threshold,
        device=device,
    )

    sample, result, metrics, mapping = run_single_recording(
        dataset=dataset,
        recording_id=recording_id,
        model=model,
        speech_selector=speech_selector,
        speech_source=config.speech_source,
        config=TuneConfig.config_stem,
        min_speech_overlap=config.min_speech_overlap,
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
    parser = argparse.ArgumentParser(description="Config-driven AMI diarization tuning.")
    parser.add_argument("--config", type=str, default="configs/tune.yml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    data = cfg["data"]
    evaluation = cfg["evaluation"]
    runtime = cfg["runtime"]
    grid = cfg["grid"]

    set_verbose(int(runtime.get("verbose", 1)))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    recordings = data["recordings"]
    ignore_overlap = bool(evaluation.get("ignore_overlap", True))
    collar = float(evaluation.get("collar", 0.25))

    config_path = Path(args.config)
    config_stem = config_path.stem
    out_csv = ROOT / runtime.get("out_csv", f"outputs/{config_stem}_tuning_results.csv")
    summary_csv = ROOT / runtime.get("summary_csv", f"outputs/{config_stem}_tuning_summary.csv")
    cache_dir = ROOT / runtime.get("cache_dir", "outputs/cache")

    configs = make_grid(grid, runtime, config_stem)

    vprint("\n=== Tuning Setup ===")
    vprint(f"config_stem:     {config_stem}")
    vprint(f"recordings:      {recordings}")
    vprint(f"device:          {device}")
    vprint(f"num_configs:     {len(configs)}")
    vprint(f"ignore_overlap:  {ignore_overlap}")
    vprint(f"collar:          {collar}")
    vprint(f"out_csv:         {out_csv}")
    vprint(f"summary_csv:     {summary_csv}")

    all_rows = []
    summary_rows = []

    for config_idx, config in enumerate(configs, start=1):
        config_dict = asdict(config)
        vprint(f"\n=== Config {config_idx}/{len(configs)} ===")
        vprint(str(config_dict))

        config_ders = []

        for recording_id in recordings:
            try:
                metrics = evaluate_config_on_recording(
                    audio_dir=data["audio_dir"],
                    annotation_dir=data["annotation_dir"],
                    recording_id=recording_id,
                    config=config,
                    device=device,
                    cache_dir=cache_dir,
                    collar=collar,
                    ignore_overlap=ignore_overlap,
                )

                config_ders.append(float(metrics["DER"]))

                all_rows.append({
                    "config_id": config_idx,
                    **config_dict,
                    **metrics,
                    "error": "",
                })

                vprint(
                    f"[Result] {recording_id} | "
                    f"DER={metrics['DER']:.4f} | "
                    f"miss={metrics['miss']:.0f} | "
                    f"FA={metrics['false_alarm']:.0f} | "
                    f"conf={metrics['confusion']:.0f}"
                )

            except Exception as exc:
                all_rows.append({
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
                })
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

        write_rows_csv(out_csv, all_rows)
        write_rows_csv(summary_csv, summary_rows)

    valid = [row for row in summary_rows if isinstance(row["mean_DER"], float)]

    if valid:
        best = min(valid, key=lambda r: r["mean_DER"])
        vprint("\n=== Best Validation Config ===")
        for key, value in best.items():
            vprint(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    else:
        vprint("\nNo successful configs completed.")

    vprint("\nDone.")


if __name__ == "__main__":
    set_seed()
    main()