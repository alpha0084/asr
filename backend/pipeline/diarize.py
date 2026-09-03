"""Stage 3 — speaker diarization (who spoke when) with pyannote."""
from ..config import DIARIZATION_MODEL, DIARIZE_DEVICE, HF_TOKEN
from ..utils import suppress_stderr

_pipeline = None


def _get_pipeline(device: str):
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Put a free HuggingFace token in .env — see README."
        )

    with suppress_stderr():                       # mute torchcodec import noise
        import torch
        from pyannote.audio import Pipeline

    license_hint = (
        "You must ACCEPT THE MODEL LICENSE once (while logged in) at:\n"
        "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        "Click 'Agree and access repository', then retry."
    )
    try:
        # huggingface_hub renamed use_auth_token -> token; support both.
        try:
            pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
        except TypeError:
            pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=HF_TOKEN)
    except Exception as e:
        raise RuntimeError(
            f"Your HF token works, but access to the model is gated.\n{license_hint}\n"
            f"(original: {type(e).__name__})"
        ) from e

    if pipe is None:
        raise RuntimeError(f"Could not load the diarization model.\n{license_hint}")

    pipe.to(torch.device(device))
    _pipeline = pipe
    return _pipeline


def diarize(waveform, sr, num_speakers: int | None = None, device: str | None = None) -> list:
    """Return [{start, end, speaker}] sorted by start time.

    Feeds an in-memory waveform dict (not a file path) so pyannote never touches
    torchcodec. num_speakers pins the count (e.g. 2 for a 1:1 call) if known.
    """
    import torch

    pipe = _get_pipeline(device or DIARIZE_DEVICE)
    wf = torch.from_numpy(waveform).unsqueeze(0)   # (1, num_samples)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    with suppress_stderr():
        output = pipe({"waveform": wf, "sample_rate": sr}, **kwargs)

    # pyannote 4.x returns DiarizeOutput(.speaker_diarization: Annotation);
    # older versions return an Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)

    turns = [
        {"start": seg.start, "end": seg.end, "speaker": spk}
        for seg, _, spk in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns
