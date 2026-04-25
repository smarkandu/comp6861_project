import torch


def events_to_frame_targets(events, speaker_to_idx, num_frames, hop_sec, num_speakers):
    """
    Convert AMI speaker events to frame-level multi-label targets.

    events format:
        {
            "start": float,
            "end": float,
            "speakers": ["A", "B"]
        }

    returns:
        targets: [T, S]
    """

    targets = torch.zeros(num_frames, num_speakers)

    for event in events:
        start = float(event["start"])
        end = float(event["end"])

        speakers = event.get("speakers", [])
        if not speakers and "speaker" in event:
            speakers = [event["speaker"]]

        start_frame = int(start / hop_sec)
        end_frame = int(end / hop_sec)

        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)

        for spk in speakers:
            if spk in speaker_to_idx:
                targets[start_frame:end_frame, speaker_to_idx[spk]] = 1.0

    return targets