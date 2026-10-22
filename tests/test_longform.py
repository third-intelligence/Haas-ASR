from itertools import pairwise

import numpy as np
import pytest

from haas_asr.longform import split_at_quiet_points

SAMPLE_RATE = 16_000


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, int(seconds * SAMPLE_RATE)).astype(np.float32)


def split(audio: np.ndarray, **kwargs) -> list[tuple[int, int]]:
    return split_at_quiet_points(audio, sample_rate=SAMPLE_RATE, **kwargs)


def test_short_audio_is_one_chunk():
    audio = noise(10)
    assert split(audio) == [(0, audio.size)]


@pytest.mark.parametrize("seconds", [30.0, 31.0, 95.5, 600.0])
def test_chunks_cover_audio_without_gaps(seconds):
    audio = noise(seconds)
    bounds = split(audio)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == audio.size
    assert all(end == next_start for (_, end), (next_start, _) in pairwise(bounds))


def test_chunk_length_is_bounded():
    audio = noise(600)
    longest = max(end - start for start, end in split(audio, chunk_seconds=30, search_seconds=2))
    assert longest <= 32 * SAMPLE_RATE


def test_cut_lands_in_silence():
    silence_start, silence_end = int(28.5 * SAMPLE_RATE), int(29.0 * SAMPLE_RATE)
    audio = noise(45)
    audio[silence_start:silence_end] = 0.0
    (_, cut), _ = split(audio)
    assert silence_start <= cut <= silence_end
