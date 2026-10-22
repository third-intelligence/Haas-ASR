"""The model behind both demos, run on a single inference thread."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time

import numpy as np
import torch

import haas_asr
from haas_asr.audio import SAMPLE_RATE


class Transcriber:
    """Loads one model and runs every call on the same worker thread.

    Calling the model from a new thread each time is 2-3x slower than calling
    it from one long-lived thread, so the web server's handlers hand their work
    to this single thread instead of calling the model themselves. Calls are
    serialized, which also keeps the two demos from competing for the GPU.
    """

    def __init__(self, model: str, device: str = "cuda"):
        # One CPU thread per core oversubscribes large machines and makes GPU
        # inference jittery; a fixed, modest count is faster.
        torch.set_num_threads(min(16, os.cpu_count() or 1))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="haas-model"
        )
        self._model = self._executor.submit(haas_asr.load_model, model, device=device).result()
        self.device_name = (
            torch.cuda.get_device_name(torch.device(device))
            if torch.device(device).type == "cuda"
            else "CPU"
        )

    def warm_up(self) -> None:
        """Run each batch shape the demos use once, so the first request is not slow."""
        noise = np.random.default_rng(0).normal(0, 0.01, 30 * SAMPLE_RATE).astype(np.float32)
        for batch in ([noise[:SAMPLE_RATE]], [noise] * 16, [noise] * 64):
            self._executor.submit(self._transcribe, batch, len(batch)).result()

    def _transcribe(self, audios: list[np.ndarray], batch_size: int) -> list[str]:
        with torch.inference_mode():
            hypotheses = self._model.transcribe(audios, batch_size=batch_size, verbose=False)
        return [hypothesis.text.strip() for hypothesis in hypotheses]

    async def transcribe(
        self, audios: list[np.ndarray], batch_size: int = 1
    ) -> tuple[list[str], float]:
        """Transcribe `audios` (16 kHz mono float32); returns the texts and seconds taken."""
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        texts = await loop.run_in_executor(self._executor, self._transcribe, audios, batch_size)
        return texts, time.perf_counter() - started
