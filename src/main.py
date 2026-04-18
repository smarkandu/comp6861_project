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

    # Other args (keep simple)
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

    args = parser.parse_args()

    run_pipeline(
        project_root=ROOT,
        audio_dir=args.audio_dir,
        annotation_dir=args.annotation_dir,
        recording_id=args.recording_id,
        debug=args.debug
    )


if __name__ == "__main__":
    main()