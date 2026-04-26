from pathlib import Path

from pathlib import Path

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