from pathlib import Path
import torch
from datasets.ami import AMIDataset
from eval.evaluation import (
    apply_mapping_to_frame_sets,
    compute_der,
    events_to_frame_sets,
    map_clusters_to_speakers,
    segments_to_frame_sets,
)
from models.baseline import BaselineDiarizer
from models.vad import SpeechBrainVAD


def save_segments(result, output_dir: Path) -> Path:
    """
    Save predicted diarization segments in a simple tab-separated text file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.recording_id}_prediction.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\n")
        for seg in result.segments:
            f.write(f"{seg.start:.2f}\t{seg.end:.2f}\t{seg.speaker}\n")

    return out_path


def print_metrics(metrics: dict) -> None:
    """
    Pretty-print the evaluation metrics so the final output is easy to read.
    """
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def run_pipeline(project_root, audio_dir, annotation_dir, recording_id, debug, vad_threshold, window_sec, hop_sec):
    print("\n=== Run Configuration ===")
    print(f"audio_dir:      {audio_dir}")
    print(f"annotation_dir: {annotation_dir}")
    print(f"recording_id:   {recording_id}")
    print(f"debug:          {debug}")
    print(f"vad_threshold:  {vad_threshold}")
    print(f"window_sec:     {window_sec}")
    print(f"hop_sec:        {hop_sec}")

    print("=== Starting Diarization Pipeline ===")

    print("\n[1/7] Loading dataset...")
    dataset = AMIDataset(
        audio_dir,
        annotation_dir,
        target_sr=16000,
    )

    print("[2/5] Selecting recording...")
    if recording_id is None:
        recordings = dataset.list_recordings()
        if not recordings:
            raise ValueError("No recordings found in the audio directory.")
        recording_id = recordings[0]
        print(f"No recording_id provided. Using default: {recording_id}")
    else:
        print(f"Using provided recording_id: {recording_id}")

    print("[3/7] Loading sample (audio + annotations)...")
    sample = dataset.load_sample(recording_id)
    print(f"Recording ID: {sample.recording_id}")
    print(f"Audio duration: {sample.duration:.2f}s")
    print(f"Number of speakers: {sample.num_speakers}")
    print(f"Speaker IDs: {sample.speakers}")
    print(f"Number of reference events: {len(sample.events)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[4/7] Initializing model on {device}...")
    model = BaselineDiarizer(
                target_sr=16000,
                window_sec=window_sec,
                hop_sec=hop_sec,
                smoothing_kernel=3,
                device=device,
                vad_threshold=vad_threshold
            )

    print("[5/7] Running diarization...")
    result = model.predict(sample)

    print("\n=== Diarization Complete ===")
    print(f"Predicted segments: {len(result.segments)}")
    print("Showing first 20 predicted segments:")
    for seg in result.segments[:20]:
        print(seg)
    if len(result.segments) > 20:
        print(f"... ({len(result.segments) - 20} more segments not shown)")

    print("[6/7] Saving predictions...")
    output_dir = f"{project_root}/outputs"
    output_dir = Path(output_dir)
    pred_file = save_segments(result, output_dir)
    print(f"Saved predictions to: {pred_file}")

    print("[7/7] Evaluating DER...")
    ref_frames = events_to_frame_sets(sample.events, sample.duration, frame_hop=0.01)
    hyp_frames = segments_to_frame_sets(result.segments, sample.duration, frame_hop=0.01)

    mapping = map_clusters_to_speakers(ref_frames, hyp_frames, ignore_overlap=True)
    print(f"Cluster mapping: {mapping}")

    hyp_frames_mapped = apply_mapping_to_frame_sets(hyp_frames, mapping)
    metrics = compute_der(ref_frames, hyp_frames_mapped, ignore_overlap=True)
    print_metrics(metrics)

