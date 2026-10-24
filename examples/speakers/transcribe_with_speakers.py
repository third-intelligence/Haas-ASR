"""Transcribe a conversation with speaker labels: Haas combined with NVIDIA's
Nemotron-3-Diarization.

The diarization model finds who speaks when; Haas transcribes each speaker's turns.
Nemotron-3-Diarization needs a newer NeMo than haas-asr pins, so this example has its
own environment (this folder's pyproject.toml). Run it from the repository root:

uv run --project examples/speakers python examples/speakers/transcribe_with_speakers.py \\
    example.wav [--words "Haas=ハース"]
"""

import argparse

import numpy as np

import haas_asr

DIARIZER = "nvidia/Nemotron-3-Diarization"
MAX_PAUSE = 1.0  # seconds: a speaker's segments closer than this are one turn
MAX_TURN = 30.0  # seconds: longer turns are cut at quiet points, as in long-form


def load_diarizer(name: str, device: str = "cuda"):
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(name, map_location=device)
    model.streaming_mode = True  # a long recording in chunks, with a speaker cache
    return model.eval()


def speaker_turns(segments: list[str], max_pause: float = MAX_PAUSE) -> list[tuple]:
    """NeMo's ``"start end speaker_N"`` segments as ``(start, end, speaker)`` turns in
    time order, joining a speaker's consecutive segments across short pauses."""
    turns = []
    for start, end, speaker in sorted(_parse(s) for s in segments):
        last = turns[-1] if turns else None
        if last and last[2] == speaker and start - last[1] < max_pause:
            turns[-1] = (last[0], max(last[1], end), speaker)
        else:
            turns.append((start, end, speaker))
    return turns


def _parse(segment: str) -> tuple[float, float, int]:
    start, end, speaker = segment.split()
    return float(start), float(end), int(speaker.rsplit("_", 1)[1])


def turn_pieces(audio: np.ndarray, turn: tuple, rate: int = haas_asr.SAMPLE_RATE) -> list:
    """A turn's audio, in pieces short enough for Haas."""
    clip = audio[int(turn[0] * rate) : int(turn[1] * rate)]
    bounds = haas_asr.split_at_quiet_points(clip, sample_rate=rate, chunk_seconds=MAX_TURN)
    return [clip[a:b] for a, b in bounds]


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes // 60):02d}:{int(minutes % 60):02d}:{seconds:04.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", help="audio file")
    parser.add_argument("--model", default="third-intelligence/Haas-ja-v1.0")
    parser.add_argument("--diarizer", default=DIARIZER)
    parser.add_argument("--words", nargs="*", default=[], help='"written form=reading,..."')
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch

    audio = haas_asr.load_audio(args.audio)
    diarizer = load_diarizer(args.diarizer)
    with torch.no_grad():
        (segments,) = diarizer.diarize(
            audio=[audio], sample_rate=haas_asr.SAMPLE_RATE, batch_size=1, verbose=False
        )
    turns = speaker_turns(segments)

    model = haas_asr.load_model(args.model)
    vocabulary = haas_asr.Vocabulary.parse(args.words)
    pieces = [turn_pieces(audio, turn) for turn in turns]
    texts = iter(
        haas_asr.transcribe(
            model,
            [piece for turn in pieces for piece in turn],
            batch_size=args.batch_size,
            vocabulary=vocabulary,
        )
    )
    for (start, _, speaker), turn in zip(turns, pieces, strict=True):
        text = "".join(next(texts) for _ in turn)
        if text:
            print(f"[{timestamp(start)}] Speaker {speaker + 1}: {text}")


if __name__ == "__main__":
    main()
