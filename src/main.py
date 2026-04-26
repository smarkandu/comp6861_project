import argparse
from pathlib import Path
import yaml

from runner import DiarizationPipeline
from debug import vprint, set_seed

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    set_seed()
    parser = argparse.ArgumentParser(description="AMI speaker diarization entry point.")

    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Configuration file path",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg_path = Path(args.config)
    cfg_stem = cfg_path.stem

    pipeline = DiarizationPipeline(
        project_root=str(ROOT),
        audio_dir=cfg["data"]["audio_dir"],
        annotation_dir=cfg["data"]["annotation_dir"],
        recording_ids=cfg["data"]["recording_ids"], #list
        debug=cfg["runtime"]["debug"],
        vad_threshold=cfg["vad"]["threshold"],
        window_sec=cfg["model"]["window_sec"],
        hop_sec=cfg["model"]["hop_sec"],
        model_type=cfg["model"]["model_type"],
        config_stem=cfg_stem,
        smoothing_kernel=cfg["model"]["smoothing_kernel"],
        collar=cfg["evaluation"]["collar"],
        ignore_overlap=cfg["evaluation"]["ignore_overlap"],
        clustering_method=cfg["cluster"]["method"],
        n_neighbors=cfg["cluster"]["n_neighbors"],
        merge_gap=cfg["segmentation"]["merge_gap"],
        min_seg_dur=cfg["segmentation"]["min_seg_dur"],
        use_embedding_cache=cfg["runtime"]["use_cache"],
        speech_source=cfg["speech"]["source"],
        min_speech_overlap=cfg["speech"]["min_speech_overlap"]
    )

    # Run Pipeline
    all_results, mean_der, std_der = pipeline.run()

    # Print Results
    vprint("\n=== Summary ===")
    for result in all_results:
        vprint(f"{result['recording_id']} → DER: {result['metrics']['DER']:.4f}")
    if mean_der is not None:
        vprint(f"\nMean DER: {mean_der:.4f}")
        vprint(f"Std DER:  {std_der:.4f}")

    return all_results, mean_der, std_der

if __name__ == "__main__":
    all_results, mean_der, std_der = main()