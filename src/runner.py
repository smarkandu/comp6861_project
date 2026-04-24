from pathlib import Path
import torch

from datasets.ami import AMIDataset
from eval.evaluation import (
    apply_mapping_to_frame_sets,
    build_collar_mask,
    compute_der,
    events_to_frame_sets,
    map_clusters_to_speakers,
    segments_to_frame_sets,
)
from models.baseline import BaselineDiarizer
from models.advanced import AdvancedDiarizer
from models.embedders import ECAPAEmbedder, WavLMEmbedder
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


def resolve_recording_audio_dir(audio_dir, recording_id):
    """
    Support both layouts:

    1) audio_dir = data/amicorpus/ES2002a/audio
    2) audio_dir = data/amicorpus and recording_id = ES2002a
       -> data/amicorpus/ES2002a/audio
    """
    audio_dir = Path(audio_dir)

    if recording_id is None:
        return audio_dir

    nested_audio_dir = audio_dir / recording_id / "audio"
    if nested_audio_dir.exists():
        return nested_audio_dir

    return audio_dir


def build_dataset(audio_dir, annotation_dir, recording_id=None, target_sr=16000):
    """
    Build AMIDataset without duplicating path logic in main.py and tune.py.
    """
    recording_audio_dir = resolve_recording_audio_dir(audio_dir, recording_id)

    return AMIDataset(
        str(recording_audio_dir),
        annotation_dir,
        target_sr=target_sr,
    )


def select_recording(dataset, recording_id):
    """
    Use the provided recording_id, or choose the first available recording.
    """
    if recording_id is not None:
        vprint(f"Using provided recording_id: {recording_id}")
        return recording_id

    recordings = dataset.list_recordings()
    if not recordings:
        raise ValueError("No recordings found in the audio directory.")

    selected = recordings[0]
    vprint(f"No recording_id provided. Using default: {selected}")
    return selected


def build_model(
    model_type,
    device,
    window_sec,
    hop_sec,
    vad_threshold,
    smoothing_kernel=1,
    cache_dir="./outputs/cache",
    use_embedding_cache=True,
    clustering_method=None,
    n_neighbors=10,
    merge_gap=0.25,
    min_seg_dur=0.75,
):
    """
    Build one diarization model.

    model_type selects the architecture/backend:
      - baseline/ecapa -> ECAPA-TDNN
      - wavlm -> WavLM
      - advanced -> legacy AdvancedDiarizer with ECAPA
    """
    normalized = model_type.lower()

    if normalized in {"baseline", "ecapa"}:
        vprint("[Model] Using ECAPA-TDNN embeddings.")
        embedder = ECAPAEmbedder(device=device)
        cls = BaselineDiarizer

    elif normalized == "wavlm":
        vprint("[Model] Using WavLM embeddings.")
        embedder = WavLMEmbedder(device=device)
        cls = BaselineDiarizer

    elif normalized == "advanced":
        vprint("[Model] Using legacy AdvancedDiarizer with ECAPA embeddings.")
        embedder = ECAPAEmbedder(device=device)
        cls = AdvancedDiarizer

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    common_kwargs = dict(
        embedder=embedder,
        target_sr=16000,
        window_sec=window_sec,
        hop_sec=hop_sec,
        smoothing_kernel=smoothing_kernel,
        device=device,
        vad_threshold=vad_threshold,
        cache_dir=cache_dir,
        use_embedding_cache=use_embedding_cache,
    )

    optional_kwargs = {
        "clustering_method": clustering_method,
        "n_neighbors": n_neighbors,
        "merge_gap": merge_gap,
        "min_seg_dur": min_seg_dur,
    }
    optional_kwargs = {k: v for k, v in optional_kwargs.items() if v is not None}

    try:
        return cls(**common_kwargs, **optional_kwargs)
    except TypeError:
        # Fallback if your current BaselineDiarizer does not accept optional args yet.
        return cls(**common_kwargs)


def evaluate_result(
    sample,
    result,
    collar=0.25,
    ignore_overlap=True,
    frame_hop=0.01,
):
    """
    Reusable DER evaluation logic.
    """
    ref_frames = events_to_frame_sets(
        sample.events,
        sample.duration,
        frame_hop=frame_hop,
    )
    hyp_frames = segments_to_frame_sets(
        result.segments,
        sample.duration,
        frame_hop=frame_hop,
    )

    mapping = map_clusters_to_speakers(
        ref_frames,
        hyp_frames,
        ignore_overlap=ignore_overlap,
    )

    hyp_frames_mapped = apply_mapping_to_frame_sets(hyp_frames, mapping)

    collar_mask = build_collar_mask(
        sample.events,
        sample.duration,
        frame_hop=frame_hop,
        collar=collar,
    )

    metrics = compute_der(
        ref_frames,
        hyp_frames_mapped,
        ignore_overlap=ignore_overlap,
        collar_mask=collar_mask,
    )

    return metrics, mapping


def run_single_recording(
    dataset,
    recording_id,
    model,
    collar=0.25,
    ignore_overlap=True,
    frame_hop=0.01,
):
    """
    Run diarization and evaluation for one recording.

    Returns:
        sample, result, metrics, mapping
    """
    sample = dataset.load_sample(recording_id)

    vprint(f"Recording ID: {sample.recording_id}")
    vprint(f"Audio duration: {sample.duration:.2f}s")
    vprint(f"Number of speakers: {sample.num_speakers}")
    vprint(f"Speaker IDs: {sample.speakers}")
    vprint(f"Number of reference events: {len(sample.events)}")

    result = model.predict(sample)

    metrics, mapping = evaluate_result(
        sample=sample,
        result=result,
        collar=collar,
        ignore_overlap=ignore_overlap,
        frame_hop=frame_hop,
    )

    return sample, result, metrics, mapping


def run_pipeline(
    project_root,
    audio_dir,
    annotation_dir,
    recording_id,
    debug,
    vad_threshold,
    window_sec,
    hop_sec,
    model_type,
    smoothing_kernel=1,
    collar=0.25,
    ignore_overlap=True,
    clustering_method=None,
    n_neighbors=10,
    merge_gap=0.25,
    min_seg_dur=0.75,
):
    vprint("\n=== Run Configuration ===")
    vprint(f"audio_dir:      {audio_dir}")
    vprint(f"annotation_dir: {annotation_dir}")
    vprint(f"recording_id:   {recording_id}")
    vprint(f"debug:          {debug}")
    vprint(f"vad_threshold:  {vad_threshold}")
    vprint(f"window_sec:     {window_sec}")
    vprint(f"hop_sec:        {hop_sec}")
    vprint(f"model_type:     {model_type}")
    vprint(f"smoothing:      {smoothing_kernel}")
    if clustering_method is not None:
        vprint(f"clustering:     {clustering_method}")

    vprint("=== Starting Diarization Pipeline ===")

    vprint("\n[1/7] Loading dataset...")
    dataset = build_dataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        recording_id=recording_id,
        target_sr=16000,
    )

    vprint("[2/7] Selecting recording...")
    recording_id = select_recording(dataset, recording_id)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    vprint(f"[3/7] Initializing model on {device}...")
    model = build_model(
        model_type=model_type,
        device=device,
        window_sec=window_sec,
        hop_sec=hop_sec,
        vad_threshold=vad_threshold,
        smoothing_kernel=smoothing_kernel,
        cache_dir=f"{project_root}/outputs/cache",
        use_embedding_cache=True,
        clustering_method=clustering_method,
        n_neighbors=n_neighbors,
        merge_gap=merge_gap,
        min_seg_dur=min_seg_dur,
    )

    vprint("[4/7] Running diarization and evaluation...")
    sample, result, metrics, mapping = run_single_recording(
        dataset=dataset,
        recording_id=recording_id,
        model=model,
        collar=collar,
        ignore_overlap=ignore_overlap,
    )

    vprint("\n=== Diarization Complete ===", 2)
    vprint(f"Predicted segments: {len(result.segments)}", 2)
    vprint("Showing first 20 predicted segments:", 2)
    for seg in result.segments[:20]:
        vprint(seg, 2)
    if len(result.segments) > 20:
        vprint(f"... ({len(result.segments) - 20} more segments not shown)", 2)

    vprint("[5/7] Saving predictions...", 2)
    output_dir = Path(project_root) / "outputs"
    pred_file = save_segments(result, output_dir)
    vprint(f"Saved predictions to: {pred_file}", 2)

    vprint("[6/7] Reporting DER...")
    vprint(f"Cluster mapping: {mapping}")
    print_metrics(metrics)

    return {
        "sample": sample,
        "result": result,
        "metrics": metrics,
        "mapping": mapping,
        "prediction_file": pred_file,
    }
