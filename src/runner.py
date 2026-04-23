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
from models.advanced import AdvancedDiarizer
from debug import vprint


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
    vprint("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            vprint(f"{key}: {value:.4f}")
        else:
            vprint(f"{key}: {value}")


def run_pipeline(project_root, audio_dir, annotation_dir, recording_id, debug, vad_threshold, window_sec, hop_sec, model_type):
    vprint("\n=== Run Configuration ===")
    vprint(f"audio_dir:      {audio_dir}")
    vprint(f"annotation_dir: {annotation_dir}")
    vprint(f"recording_id:   {recording_id}")
    vprint(f"debug:          {debug}")
    vprint(f"vad_threshold:  {vad_threshold}")
    vprint(f"window_sec:     {window_sec}")
    vprint(f"hop_sec:        {hop_sec}")
    vprint(f"model_type:     {model_type}")

    vprint("=== Starting Diarization Pipeline ===")

    vprint("\n[1/7] Loading dataset...")
    dataset = AMIDataset(
        audio_dir,
        annotation_dir,
        target_sr=16000,
    )

    vprint("[2/5] Selecting recording...")
    if recording_id is None:
        recordings = dataset.list_recordings()
        if not recordings:
            raise ValueError("No recordings found in the audio directory.")
        recording_id = recordings[0]
        vprint(f"No recording_id provided. Using default: {recording_id}")
    else:
        vprint(f"Using provided recording_id: {recording_id}")

    vprint("[3/7] Loading sample (audio + annotations)...")
    sample = dataset.load_sample(recording_id)
    vprint(f"Recording ID: {sample.recording_id}")
    vprint(f"Audio duration: {sample.duration:.2f}s")
    vprint(f"Number of speakers: {sample.num_speakers}")
    vprint(f"Speaker IDs: {sample.speakers}")
    vprint(f"Number of reference events: {len(sample.events)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vprint(f"[4/7] Initializing model on {device}...")
    model = None
    if model_type == "baseline":
        model = BaselineDiarizer(
            target_sr=16000,
            window_sec=window_sec,
            hop_sec=hop_sec,
            smoothing_kernel=1,
            device=device,
            vad_threshold=vad_threshold,
        )
    elif model_type == "advanced":
        model = AdvancedDiarizer(
            target_sr=16000,
            window_sec=window_sec,
            hop_sec=hop_sec,
            smoothing_kernel=1,
            device=device,
            vad_threshold=vad_threshold,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    vprint("[5/7] Running diarization...")
    result = model.predict(sample)

    vprint("\n=== Diarization Complete ===")
    vprint(f"Predicted segments: {len(result.segments)}")
    vprint("Showing first 20 predicted segments:")
    for seg in result.segments[:20]:
        vprint(seg)
    if len(result.segments) > 20:
        vprint(f"... ({len(result.segments) - 20} more segments not shown)")

    vprint("[6/7] Saving predictions...")
    output_dir = f"{project_root}/outputs"
    output_dir = Path(output_dir)
    pred_file = save_segments(result, output_dir)
    vprint(f"Saved predictions to: {pred_file}")

    vprint("[7/7] Evaluating DER...")
    ref_frames = events_to_frame_sets(sample.events, sample.duration, frame_hop=0.01)
    hyp_frames = segments_to_frame_sets(result.segments, sample.duration, frame_hop=0.01)

    mapping = map_clusters_to_speakers(ref_frames, hyp_frames, ignore_overlap=True)
    vprint(f"Cluster mapping: {mapping}")

    hyp_frames_mapped = apply_mapping_to_frame_sets(hyp_frames, mapping)
    metrics = compute_der(ref_frames, hyp_frames_mapped, ignore_overlap=True)
    print_metrics(metrics)

