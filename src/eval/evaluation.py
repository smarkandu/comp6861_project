"""
evaluation.py

Evaluation utilities for speaker diarization baseline.

References:
[1] Desplanques et al., "ECAPA-TDNN...", Interspeech 2020.
[2] Dawalatabad et al., "ECAPA-TDNN Embeddings for Speaker Diarization", 2021.
[3] von Luxburg, "A Tutorial on Spectral Clustering", 2007.
[4] Anguera et al., "Speaker Diarization: A Review...", IEEE TASLP, 2012.
    (Used for DER definition and overlap-aware scoring)

Key design choice:
- Frame-level evaluation with speaker SETS per frame
- This allows easy extension to overlapping speech later
"""

import numpy as np
from collections import Counter

# Special tokens
SILENCE = "__silence__"


# --------------------------------------------------
# 1. Reference → Frame Sets
# --------------------------------------------------

def events_to_frame_sets(events, duration, frame_hop=0.01):
    """
    Convert ground-truth events into frame-level speaker sets.

    Each frame stores a SET of active speakers:
        set()        → silence
        {"A"}        → single speaker
        {"A","B"}    → overlap

    Why sets?
    - Overlap-ready design (important for future extension)
    - Matches DER definition using speaker counts [4]

    Parameters:
        events: list of dicts with {"start", "end", "speakers"}
        duration: total audio duration
        frame_hop: frame step (seconds)

    Returns:
        List[Set[str]]
    """
    n_frames = int(np.ceil(duration / frame_hop))
    frames = [set() for _ in range(n_frames)]

    for ev in events:
        start_idx = int(np.floor(ev["start"] / frame_hop))
        end_idx = int(np.ceil(ev["end"] / frame_hop))
        speakers = set(ev["speakers"])  # may contain multiple speakers

        for i in range(max(0, start_idx), min(n_frames, end_idx)):
            frames[i].update(speakers)

    return frames


# --------------------------------------------------
# 2. Hypothesis → Frame Sets
# --------------------------------------------------

def segments_to_frame_sets(segments, duration, frame_hop=0.01):
    """
    Convert predicted segments into frame-level speaker sets.

    Current baseline:
        - Each segment has ONE speaker (cluster)
    Future extension:
        - Multiple speakers per frame → handled naturally

    This aligns with diarization pipelines using embeddings + clustering [2].

    Parameters:
        segments: List[DiarizationSegment]
        duration: audio duration
        frame_hop: frame step

    Returns:
        List[Set[str]]
    """
    n_frames = int(np.ceil(duration / frame_hop))
    frames = [set() for _ in range(n_frames)]

    for seg in segments:
        start_idx = int(np.floor(seg.start / frame_hop))
        end_idx = int(np.ceil(seg.end / frame_hop))

        for i in range(max(0, start_idx), min(n_frames, end_idx)):
            frames[i].add(seg.speaker)

    return frames


# --------------------------------------------------
# 3. Cluster → Speaker Mapping
# --------------------------------------------------

def map_clusters_to_speakers(ref_frames, hyp_frames, ignore_overlap=True):
    """
    Map predicted cluster labels (cluster_0, cluster_1, ...)
    to true speaker IDs (A, B, ...).

    Uses majority voting:
    - For each predicted cluster, assign the most frequent true speaker

    This is consistent with standard clustering evaluation practice [3].

    Parameters:
        ignore_overlap:
            If True → skip frames with multiple reference speakers

    Returns:
        Dict[str, str]
    """
    votes = {}

    for ref_set, hyp_set in zip(ref_frames, hyp_frames):
        # Skip overlap frames if doing baseline evaluation
        if ignore_overlap and len(ref_set) > 1:
            continue

        # Only consider single-speaker frames
        if len(ref_set) != 1 or len(hyp_set) != 1:
            continue

        ref_spk = next(iter(ref_set))
        hyp_spk = next(iter(hyp_set))

        votes.setdefault(hyp_spk, []).append(ref_spk)

    mapping = {}
    for cluster, refs in votes.items():
        mapping[cluster] = Counter(refs).most_common(1)[0][0]

    return mapping


def apply_mapping_to_frame_sets(hyp_frames, mapping):
    """
    Replace cluster labels with mapped speaker IDs.

    This step is critical:
    - Clustering labels are arbitrary
    - Must align with ground truth before scoring
    """
    out = []
    for hyp_set in hyp_frames:
        mapped = {mapping.get(spk, spk) for spk in hyp_set}
        out.append(mapped)
    return out


# --------------------------------------------------
# 4. DER Computation (Overlap-Ready)
# --------------------------------------------------

def compute_der(ref_frames, hyp_frames, ignore_overlap=True):
    """
    Compute Diarization Error Rate (DER).

    Based on [4]:
        MISS = max(0, R - H)
        FA   = max(0, H - R)
        CONF = min(R, H) - C

    Where:
        R = number of reference speakers
        H = number of hypothesis speakers
        C = correctly matched speakers

    Parameters:
        ignore_overlap:
            True  → baseline evaluation
            False → full overlap-aware DER

    Returns:
        dict with DER and components
    """
    miss = 0
    false_alarm = 0
    confusion = 0
    ref_total = 0
    scored_frames = 0

    for ref_set, hyp_set in zip(ref_frames, hyp_frames):

        # Skip overlap frames for baseline (Option A)
        if ignore_overlap and len(ref_set) > 1:
            continue

        scored_frames += 1

        R = len(ref_set)
        H = len(hyp_set)
        C = len(ref_set.intersection(hyp_set))

        # Total reference speaker time
        ref_total += R

        # DER components
        miss += max(0, R - H)
        false_alarm += max(0, H - R)
        confusion += min(R, H) - C

    der = (miss + false_alarm + confusion) / ref_total if ref_total > 0 else 0.0

    return {
        "DER": der,
        "miss": miss,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "ref_total": ref_total,
        "scored_frames": scored_frames,
    }