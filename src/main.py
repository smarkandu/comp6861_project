import argparse
from pathlib import Path
from runner import run_pipeline


# Get current file location (same as your runner)
ROOT = Path(__file__).resolve().parent

# From this position, obtain project folder
# <project_root>/src/main.py
ROOT = ROOT.parent
ROOT = str(ROOT)


def main():
    parser = argparse.ArgumentParser(description="AMI speaker diarization entry point.")

    # Debug flag (default = True)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug outputs (default: True)"
    )

    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable debug outputs"
    )

    parser.add_argument(
        "--recording-id",
        type=str,
        default=None,
        help="Recording ID (default: first available)"
    )

    parser.add_argument(
        "--audio-dir",
        type=str,
        default=f"{ROOT}/data/amicorpus/ES2002a/audio",
        help="Path to audio directory"
    )

    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=f"{ROOT}/data/ami_public_manual_1.6.2",
        help="Path to annotation directory"
    )

    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=8e-5,
        help="Energy threshold for silence filtering (default: 8e-5)"
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=3,
        help="Sliding window length in seconds (default: 3)"
    )

    parser.add_argument(
        "--hop-sec",
        type=float,
        default=1.5,
        help="Sliding window hop in seconds (default: 1.5)"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="ecapa",
        choices=["baseline", "ecapa", "wavlm", "advanced"],
        help=(
            "Which diarization architecture to use. "
            "'baseline' is a legacy alias for 'ecapa'. "
            "'wavlm' uses WavLM embeddings. "
            "'advanced' keeps the old spectral-clustering AdvancedDiarizer path."
        )
    )

    args = parser.parse_args()

    run_pipeline(
        project_root=ROOT,
        audio_dir=args.audio_dir,
        annotation_dir=args.annotation_dir,
        recording_id=args.recording_id,
        debug=args.debug,
        vad_threshold=args.vad_threshold,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
