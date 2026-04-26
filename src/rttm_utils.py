from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

def segments_to_events(segments, mapping=None):
    events = []

    for seg in segments:
        speaker = seg.speaker

        if mapping is not None:
            speaker = mapping.get(speaker, speaker)

        events.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "speakers": [speaker],
        })

    return events


def write_reference_rttm(events, recording_id: str, output_path: str):
    """
    Convert AMI ground-truth speaker events into RTTM format.

    Expected event format:
        {
            "start": float,
            "end": float,
            "speakers": list[str]
        }

    RTTM format:
        SPEAKER <recording_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

    If multiple speakers are active in the same event, write one RTTM line per speaker.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            start = float(event["start"])
            end = float(event["end"])
            duration = end - start

            if duration <= 0:
                continue

            speakers = event.get("speakers", [])

            if not speakers:
                continue

            for speaker in speakers:
                line = (
                    f"SPEAKER {recording_id} 1 "
                    f"{start:.3f} {duration:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )
                f.write(line)

def load_rttm(rttm_path):
    segments = []
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append((start, start + dur, speaker))
    return segments


def group_by_speaker(segments):
    grouped = defaultdict(list)
    for start, end, speaker in segments:
        grouped[speaker].append((start, end))
    return grouped


def plot_rttm(ax, grouped_segments, title):
    for i, (speaker, segs) in enumerate(sorted(grouped_segments.items())):
        for start, end in segs:
            ax.barh(i, end - start, left=start)
    ax.set_yticks(range(len(grouped_segments)))
    ax.set_yticklabels(sorted(grouped_segments.keys()))
    ax.set_title(title)
    ax.set_xlabel("Time (s)")


def compare_rttm(ref_path, hyp_path):
    ref = load_rttm(ref_path)
    hyp = load_rttm(hyp_path)

    ref_grouped = group_by_speaker(ref)
    hyp_grouped = group_by_speaker(hyp)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    plot_rttm(axes[0], ref_grouped, "Reference RTTM")
    plot_rttm(axes[1], hyp_grouped, "Predicted RTTM")

    plt.tight_layout()
    plt.show()