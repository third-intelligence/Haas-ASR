import numpy as np
import pytest

from demos.segmenter import PhraseSegmenter, SegmenterConfig

SAMPLE_RATE = 16_000
FRAME = SAMPLE_RATE // 10  # 100 ms frames, as the page sends them


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.3, 0.3, int(seconds * SAMPLE_RATE)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def feed(segmenter: PhraseSegmenter, audio: np.ndarray) -> None:
    for start in range(0, audio.size, FRAME):
        segmenter.push(audio[start : start + FRAME])


def test_leading_silence_is_dropped():
    segmenter = PhraseSegmenter()
    feed(segmenter, np.concatenate([silence(2.0), speech(1.0)]))
    assert segmenter.start_seconds == pytest.approx(2.0)
    assert segmenter.audio.size == SAMPLE_RATE


def test_short_fragment_is_not_readable():
    segmenter = PhraseSegmenter()
    feed(segmenter, speech(0.5))
    assert not segmenter.readable()
    feed(segmenter, speech(0.5, seed=1))
    assert segmenter.readable()


def test_pause_ends_a_phrase():
    segmenter = PhraseSegmenter()
    feed(segmenter, speech(2.0))
    assert not segmenter.ended_by_pause()
    feed(segmenter, silence(0.7))
    assert segmenter.ended_by_pause()
    phrase = segmenter.take_phrase()
    assert phrase.start_seconds == 0.0
    assert phrase.audio.size == int(2.7 * SAMPLE_RATE)
    assert segmenter.audio.size == 0


def test_noise_followed_by_a_pause_is_dropped():
    segmenter = PhraseSegmenter()
    feed(segmenter, np.concatenate([speech(0.1), silence(1.0)]))
    assert segmenter.audio.size == 0
    assert not segmenter.ended_by_pause()


def test_long_phrase_is_cut_at_the_quietest_point_and_nothing_is_lost():
    config = SegmenterConfig(max_phrase_seconds=14.0, cut_search_seconds=3.0)
    segmenter = PhraseSegmenter(config)
    # A 0.3 s dip (quiet, but too short to end the phrase) 2 s before the limit.
    dip = np.full(int(0.3 * SAMPLE_RATE), 0.001, dtype=np.float32)
    audio = np.concatenate([speech(11.7), dip, speech(2.0, seed=1)])
    feed(segmenter, audio)
    assert segmenter.too_long()
    phrase = segmenter.take_phrase()
    cut = phrase.audio.size
    assert 11.7 * SAMPLE_RATE <= cut <= 12.0 * SAMPLE_RATE
    assert segmenter.start_seconds == pytest.approx(cut / SAMPLE_RATE)
    np.testing.assert_array_equal(np.concatenate([phrase.audio, segmenter.audio]), audio)
