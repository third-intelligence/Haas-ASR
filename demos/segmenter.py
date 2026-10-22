"""Phrase segmentation for the real-time demo.

Haas-ja-v1.0 is an offline model: it reads a whole utterance at once. The
real-time demo makes it feel live by re-reading the phrase being spoken every
few hundred milliseconds. This module decides what a "phrase" is:

- A phrase starts at the first speech after silence; silence before it is dropped.
- It ends after `pause_seconds` of silence, or once it reaches `max_phrase_seconds`
  of audio. A length cut is placed at the quietest point of the phrase's last
  `cut_search_seconds`, and the audio after it starts the next phrase, so nothing
  is lost and every read covers the whole open phrase.
- It is worth reading once it holds `min_audio_seconds` of audio, including at least
  `min_speech_seconds` of speech. Shorter fragments make the model guess.

Speech and silence are told apart by the RMS level of each incoming frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from haas_asr.audio import SAMPLE_RATE


@dataclass(frozen=True)
class SegmenterConfig:
    pause_seconds: float = 0.7
    max_phrase_seconds: float = 14.0
    cut_search_seconds: float = 3.0
    min_audio_seconds: float = 0.8
    min_speech_seconds: float = 0.3
    # RMS of a float32 frame below which it counts as silence (about -38 dBFS).
    silence_rms: float = 0.012


@dataclass(frozen=True)
class Phrase:
    """A finished phrase: its audio and where it starts in the session, in seconds."""

    audio: np.ndarray
    start_seconds: float


class PhraseSegmenter:
    """Accumulates audio frames and splits them into phrases."""

    def __init__(self, config: SegmenterConfig | None = None, sample_rate: int = SAMPLE_RATE):
        self.config = config or SegmenterConfig()
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._length = 0  # samples in the open phrase
        self._speech = 0  # speech samples in the open phrase
        self._silence = 0  # trailing silent samples
        self._received = 0  # samples received in the session
        self._start = 0  # session sample where the open phrase starts

    # -- input ------------------------------------------------------------------

    def push(self, frame: np.ndarray) -> None:
        """Add one frame of 16 kHz mono float32 audio."""
        silent = float(np.sqrt(np.mean(np.square(frame)))) < self.config.silence_rms
        if silent and self._speech == 0:
            # Nothing said yet: drop the silence so the phrase starts at speech.
            self._received += frame.size
            return
        if self._length == 0:
            self._start = self._received
        self._frames.append(frame)
        self._length += frame.size
        self._received += frame.size
        if silent:
            self._silence += frame.size
            if (
                self._silence >= self.config.pause_seconds * self.sample_rate
                and not self.readable()
            ):
                # A short noise followed by a pause is not a phrase: drop it.
                self._reset(self._received)
        else:
            self._silence = 0
            self._speech += frame.size

    # -- state ------------------------------------------------------------------

    @property
    def audio(self) -> np.ndarray:
        """The open phrase so far."""
        if len(self._frames) > 1:
            self._frames = [np.concatenate(self._frames)]
        return self._frames[0] if self._frames else np.zeros(0, dtype=np.float32)

    @property
    def start_seconds(self) -> float:
        return self._start / self.sample_rate

    def readable(self) -> bool:
        """Whether the open phrase is long enough to be worth reading."""
        return (
            self._length >= self.config.min_audio_seconds * self.sample_rate
            and self._speech >= self.config.min_speech_seconds * self.sample_rate
        )

    def ended_by_pause(self) -> bool:
        return self.readable() and self._silence >= self.config.pause_seconds * self.sample_rate

    def too_long(self) -> bool:
        return self._length >= self.config.max_phrase_seconds * self.sample_rate

    # -- output -----------------------------------------------------------------

    def take_phrase(self) -> Phrase:
        """Close the open phrase: at a pause, all of it; when too long, up to a quiet cut."""
        audio, start = self.audio, self._start
        if not self.too_long():
            self._reset(self._received)
            return Phrase(audio, start / self.sample_rate)
        cut = self._quiet_cut(audio)
        rest = audio[cut:]
        self._reset(start + cut)
        if rest.size:
            self._frames = [rest]
            self._length = self._speech = rest.size
        return Phrase(audio[:cut], start / self.sample_rate)

    def _reset(self, start: int) -> None:
        self._frames, self._length, self._speech, self._silence = [], 0, 0, 0
        self._start = start

    def _quiet_cut(self, audio: np.ndarray) -> int:
        """Centre of the lowest-energy 100 ms window in the last `cut_search_seconds`."""
        window = self.sample_rate // 10
        first = max(0, audio.size - int(self.config.cut_search_seconds * self.sample_rate))
        span = audio[first:].astype(np.float64)
        if span.size <= window:
            return audio.size
        energy = np.cumsum(np.concatenate([[0.0], span**2]))
        offset = int(np.argmin(energy[window:] - energy[:-window]))
        return first + offset + window // 2
