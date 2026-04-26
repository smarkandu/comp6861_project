from vad.BaseSpeechRegionSelector import BaseSpeechRegionSelector
from vad.VADSpeechRegionSelector import VADSpeechRegionSelector
from vad.OracleSpeechRegionSelector import OracleSpeechRegionSelector
from vad.SpeechBrainVAD import SpeechBrainVAD


# Helper function
def build_speech_region_selector(
    source: str,
    vad_threshold: float = 8e-5,
    device: str = "cpu",
) -> BaseSpeechRegionSelector:
    source = source.lower()

    if source == "oracle":
        return OracleSpeechRegionSelector()

    if source == "speechbrain_vad":
        return VADSpeechRegionSelector(
            SpeechBrainVAD(device=device)
        )

    raise ValueError(
        f"Unknown speech source: {source}. "
        "Expected one of: oracle, energy_vad, speechbrain_vad"
    )