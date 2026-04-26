import torch
from pathlib import Path

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
from models.embedders.ECAPAEmbedder import ECAPAEmbedder
from models.embedders.WavLMEmbedder import WavLMEmbedder
from vad.BaseVAD import filter_windows_by_regions
from vad.SpeechRegionSelectorFactory import build_speech_region_selector
from utils.debug import vprint, set_debug, set_seed
from utils.rttm_utils import write_reference_rttm, segments_to_events, compare_rttm
import numpy as np


class DiarizationPipeline:
    def __init__(
        self,
        project_root,
        audio_dir,
        annotation_dir,
        recording_ids,
        debug,
        vad_threshold,
        window_sec,
        hop_sec,
        model_type,
        config_stem,
        smoothing_kernel=1,
        collar=0.25,
        ignore_overlap=True,
        clustering_method=None,
        n_neighbors=10,
        merge_gap=0.25,
        min_seg_dur=0.75,
        cache_dir=None,
        use_embedding_cache=True,
        speech_source="oracle",
        min_speech_overlap=0.5,
    ):
        self.project_root = project_root
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.recording_ids = recording_ids
        self.debug = debug
        self.vad_threshold = vad_threshold
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.model_type = model_type
        self.config_stem = config_stem
        self.smoothing_kernel = smoothing_kernel
        self.collar = collar
        self.ignore_overlap = ignore_overlap
        self.clustering_method = clustering_method
        self.n_neighbors = n_neighbors
        self.merge_gap = merge_gap
        self.min_seg_dur = min_seg_dur
        self.cache_dir = cache_dir or f"{project_root}/outputs/cache"
        self.use_embedding_cache = use_embedding_cache
        self.speech_source = speech_source
        self.min_speech_overlap = min_speech_overlap

    def run(self):
        set_seed()
        set_debug(self.debug)

        vprint("\n=== Run Configuration ===")
        vprint(f"config stem:        {self.config_stem}")
        vprint(f"audio_dir:        {self.audio_dir}")
        vprint(f"annotation_dir:   {self.annotation_dir}")
        vprint(f"recording_ids:     {self.recording_ids}")
        vprint(f"debug:            {self.debug}")
        vprint(f"use_cache:        {self.use_embedding_cache}")

        vprint(f"\n--- Model ---")
        vprint(f"model_type:       {self.model_type}")
        vprint(f"window_sec:       {self.window_sec}")
        vprint(f"hop_sec:          {self.hop_sec}")
        vprint(f"smoothing:        {self.smoothing_kernel}")

        vprint(f"\n--- Clustering ---")
        vprint(f"method:           {self.clustering_method}")
        vprint(f"n_neighbors:      {self.n_neighbors}")
        vprint(f"merge_gap:        {self.merge_gap}")
        vprint(f"min_seg_dur:      {self.min_seg_dur}")

        vprint(f"\n--- Speech ---")
        vprint(f"speech_source:    {self.speech_source}")
        vprint(f"min_overlap:      {self.min_speech_overlap}")
        vprint(f"vad_threshold:    {self.vad_threshold}")
        vprint("=== Starting Diarization Pipeline ===")

        vprint("\n[1/7] Loading dataset...")
        dataset, _ = build_dataset(
            audio_dir=self.audio_dir,
            annotation_dir=self.annotation_dir,
            recording_ids=self.recording_ids,
            target_sr=16000,
        )

        vprint("[2/7] Selecting recordings...")
        if self.recording_ids is None:
            vprint("[INFO] No recording_id provided → running ALL recordings")
            recording_ids = dataset.list_recordings()
        else:
            recording_ids = self.recording_ids

        print("recording ids: ", recording_ids)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        vprint(f"[3/7] Initializing model on {device}...")
        model = build_model(
            model_type=self.model_type,
            device=device,
            window_sec=self.window_sec,
            hop_sec=self.hop_sec,
            vad_threshold=self.vad_threshold,
            smoothing_kernel=self.smoothing_kernel,
            cache_dir=self.cache_dir,
            use_embedding_cache=self.use_embedding_cache,
            clustering_method=self.clustering_method,
            n_neighbors=self.n_neighbors,
            merge_gap=self.merge_gap,
            min_seg_dur=self.min_seg_dur,
        )

        speech_selector = build_speech_region_selector(
            source=self.speech_source,
            vad_threshold=self.vad_threshold,
            device=device,
        )

        vprint("[4/7] Running diarization and evaluation...")
        all_results = []

        for recording_id in recording_ids:
            vprint(f"\n=== Processing {recording_id} ===")
            sample, result, metrics, mapping = run_single_recording(
                dataset=dataset,
                recording_id=recording_id,
                model=model,
                speech_selector=speech_selector,
                speech_source=self.speech_source,
                config= self.config_stem,
                min_speech_overlap=self.min_speech_overlap,
                collar=self.collar,
                ignore_overlap=self.ignore_overlap,
                generate_output=True
            )

            vprint("\n=== Diarization Complete ===", 2)
            vprint(f"Predicted segments: {len(result.segments)}", 2)
            vprint("Showing first 20 predicted segments:", 2)

            for seg in result.segments[:20]:
                vprint(seg, 2)

            if len(result.segments) > 20:
                vprint(f"... ({len(result.segments) - 20} more segments not shown)", 2)

            vprint("[5/7] Saving predictions...")
            output_dir = Path(self.project_root) / "outputs"
            pred_file = save_segments(result, output_dir)
            vprint(f"Saved predictions to: {pred_file}", 2)

            vprint("[6/7] Reporting DER...")
            vprint(f"Cluster mapping: {mapping}")
            print_metrics(metrics)

            all_results.append({
                "recording_id": recording_id,
                "sample": sample,
                "result": result,
                "metrics": metrics,
                "mapping": mapping,
                "prediction_file": pred_file,
            })

        # obtain all DERS
        ders = [r["metrics"]["DER"] for r in all_results if "DER" in r["metrics"]]

        # Calculate the mean DER and the std deviation for DER
        if ders:
            mean_der = float(np.mean(ders))
            std_der = float(np.std(ders))
        else:
            mean_der = None
            std_der = None

        return all_results, mean_der, std_der

def save_segments(result, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.recording_id}_prediction.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("start\tend\tspeaker\n")
        for seg in result.segments:
            f.write(f"{seg.start:.2f}\t{seg.end:.2f}\t{seg.speaker}\n")

    return out_path


def print_metrics(metrics: dict) -> None:
    vprint("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            vprint(f"{key}: {value:.4f}")
        else:
            vprint(f"{key}: {value}")


def build_dataset(audio_dir, annotation_dir, target_sr, recording_ids):
    dataset = AMIDataset(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        target_sr=target_sr,
    )

    if recording_ids is None:
        recording_ids = dataset.list_recordings()
    elif isinstance(recording_ids, str):
        recording_ids = [recording_ids]
    elif isinstance(recording_ids, list):
        pass
    else:
        raise ValueError(
            "recording_ids must be None, a string, or a list of strings"
        )

    return dataset, recording_ids

def select_recording(dataset, recording_id):
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
    model_type = model_type.lower()

    if model_type in {"baseline", "ecapa"}:
        vprint("[Model] Using ECAPA-TDNN embeddings.")
        embedder = ECAPAEmbedder(device=device)

    elif model_type == "wavlm":
        vprint("[Model] Using WavLM embeddings.")
        embedder = WavLMEmbedder(device=device)

    elif model_type == "advanced":
        vprint("[Model] Using AdvancedDiarizer with WavLM embeddings.")
        # embedder = ECAPAEmbedder(device=device)
        embedder = WavLMEmbedder(device=device)
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
        clustering_method=clustering_method,
        n_neighbors=n_neighbors,
        merge_gap=merge_gap,
        min_seg_dur=min_seg_dur,
    )

    # Generates Model
    if model_type in {"baseline", "ecapa"}:
        return BaselineDiarizer(**common_kwargs)
    elif model_type == "wavlm":
        return AdvancedDiarizer(**common_kwargs)
    elif model_type == "advanced":
        return AdvancedDiarizer(**common_kwargs)
    return None

def evaluate_result(
    sample,
    result,
    collar=0.25,
    ignore_overlap=True,
    frame_hop=0.01,
):
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
    speech_selector,
    speech_source,
    config,
    min_speech_overlap=0.5,
    collar=0.25,
    ignore_overlap=True,
    frame_hop=0.01,
    generate_output=False
):
    sample = dataset.load_sample(recording_id)

    vprint(f"Recording ID: {sample.recording_id}")
    vprint(f"Audio duration: {sample.duration:.2f}s")
    vprint(f"Number of speakers: {sample.num_speakers}")
    vprint(f"Speaker IDs: {sample.speakers}")
    vprint(f"Number of reference events: {len(sample.events)}")

    model.set_num_speakers(sample.num_speakers)

    speech_regions = speech_selector.get_speech_regions(sample)
    vprint(f"Number of speech regions: {len(speech_regions)}")

    audio = model._prepare_audio(sample.audio, sample.sr)

    source = speech_source.lower()

    if source == "oracle":
        vprint("[Speech] Creating windows from oracle speech regions.")
        windows, times = model._make_windows_from_regions(audio, speech_regions)

    elif source in {"vad", "speechbrain_vad", "energy_vad"}:
        vprint("[Speech] Creating full sliding windows, then filtering by VAD regions.")
        windows, times = model._make_sliding_windows(audio)

        windows, times = filter_windows_by_regions(
            windows=windows,
            times=times,
            speech_regions=speech_regions,
            min_speech_overlap=min_speech_overlap,
        )

    else:
        raise ValueError(
            f"Unknown speech_source: {speech_source}. "
            "Expected oracle, vad, speechbrain_vad, or energy_vad."
        )

    vprint(f"[Speech] Windows after speech selection: {len(windows)}")

    result = model.predict_windows(
        recording_id=sample.recording_id,
        windows=windows,
        times=times,
    )

    # Write RTTM
    result_events = segments_to_events(result.segments)

    metrics, mapping = evaluate_result(
        sample=sample,
        result=result,
        collar=collar,
        ignore_overlap=ignore_overlap,
        frame_hop=frame_hop,
    )
    mapped_events = segments_to_events(result.segments, mapping=mapping)

    if generate_output:
        ref_path = f"outputs/rttm/{config}/{recording_id}_ref.rttm"
        hyp_path = f"outputs/rttm/{config}/{recording_id}_hyp_raw.rttm"
        hyp_map_path = f"outputs/rttm/{config}/{recording_id}_hyp_map.rttm"
        compare_rttm_path = f"outputs/rttm/{config}/{recording_id}.jpg"
        write_reference_rttm(sample.events, recording_id, ref_path)
        write_reference_rttm(result_events, recording_id, hyp_path)
        write_reference_rttm(mapped_events, recording_id, hyp_map_path)
        compare_rttm(recording_id, ref_path, hyp_map_path, compare_rttm_path)

    return sample, result, metrics, mapping