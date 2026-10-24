"""examples/speakers: turning the diarizer's segments into speaker turns (no models)."""

import importlib.util
from pathlib import Path

import numpy as np

EXAMPLE = Path(__file__).parents[1] / "examples" / "speakers" / "transcribe_with_speakers.py"
spec = importlib.util.spec_from_file_location("transcribe_with_speakers", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_segments_become_turns_in_time_order():
    segments = [
        "4.000 6.000 speaker_1",
        "0.000 1.500 speaker_0",
        "2.000 3.500 speaker_0",  # 0.5 s pause: same turn
        "7.500 9.000 speaker_1",  # 1.5 s pause: a new turn
    ]
    assert example.speaker_turns(segments) == [
        (0.0, 3.5, 0),
        (4.0, 6.0, 1),
        (7.5, 9.0, 1),
    ]


def test_a_long_turn_is_cut_into_pieces_haas_can_read():
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, 100 * 16000).astype(np.float32)
    pieces = example.turn_pieces(audio, (10.0, 80.0, 0))
    assert sum(p.size for p in pieces) == 70 * 16000
    assert max(p.size for p in pieces) <= 32 * 16000
