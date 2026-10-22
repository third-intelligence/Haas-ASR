"""Real-time demo: transcribe microphone audio as it arrives.

The page sends 16 kHz mono audio in short frames, each prefixed with the page's
own timestamp (8-byte little-endian float64, `performance.now()` in ms). The
server re-reads the open phrase about every `READ_INTERVAL` seconds and sends
the text back with the timestamp of the newest frame that read included, so the
page can measure the delay end to end on a single clock.

    page -> server  {"type": "start"}, audio frames, {"type": "stop"}
    server -> page  {"type": "update", "phrases": [{"start", "text"}],
                     "pending": {"start", "text"} | null,
                     "latency": {"stamp", "model_ms"} | null}
                    {"type": "stopped"}
                    {"type": "error", "message"}

Settled phrases never change; the pending phrase is re-read as audio arrives.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .inference import Transcriber
from .segmenter import PhraseSegmenter

READ_INTERVAL = 0.25  # seconds between reads of the open phrase


class Session:
    """One recording: incoming frames, the segmenter and the text so far."""

    def __init__(self, websocket: WebSocket, transcriber: Transcriber):
        self.websocket = websocket
        self.transcriber = transcriber
        self.segmenter = PhraseSegmenter()
        self.phrases: list[dict] = []
        self.pending: dict | None = None
        self.stamp: float | None = None  # page timestamp of the newest frame
        self.fresh = False  # audio arrived since the last read
        self.running = True

    def push(self, message: bytes) -> None:
        """Add one frame: an 8-byte timestamp, then 16-bit little-endian PCM."""
        stamp, payload = message[:8], message[8:]
        payload = payload[: len(payload) // 2 * 2]  # whole samples only
        if len(stamp) < 8 or not payload:
            return
        (self.stamp,) = struct.unpack("<d", stamp)
        pcm = np.frombuffer(payload, dtype="<i2")
        self.segmenter.push(pcm.astype(np.float32) / 32768.0)
        self.fresh = True

    async def send(self, latency: dict | None = None) -> None:
        await self.websocket.send_json(
            {"type": "update", "phrases": self.phrases, "pending": self.pending, "latency": latency}
        )

    async def settle(self) -> None:
        """Read the finished phrase once more and freeze its text."""
        phrase = self.segmenter.take_phrase()
        (text,), _ = await self.transcriber.transcribe([phrase.audio])
        if text:
            self.phrases.append({"start": phrase.start_seconds, "text": text})
        self.pending = None
        await self.send()

    async def read_open_phrase(self) -> None:
        stamp, start = self.stamp, self.segmenter.start_seconds
        self.fresh = False
        (text,), seconds = await self.transcriber.transcribe([self.segmenter.audio])
        self.pending = {"start": start, "text": text} if text else None
        await self.send({"stamp": stamp, "model_ms": seconds * 1000})

    async def run(self) -> None:
        try:
            await self._loop()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # e.g. out of GPU memory: tell the page instead of going quiet
            self.running = False
            await self.websocket.send_json({"type": "error", "message": str(error)})

    async def _loop(self) -> None:
        last_read = 0.0
        while self.running:
            if self.segmenter.too_long() or self.segmenter.ended_by_pause():
                await self.settle()
            elif (
                self.fresh
                and self.segmenter.readable()
                and time.perf_counter() - last_read >= READ_INTERVAL
            ):
                last_read = time.perf_counter()
                await self.read_open_phrase()
            else:
                await asyncio.sleep(0.02)

    async def finish(self) -> None:
        """Settle what is left after the page stops, then confirm."""
        if self.segmenter.readable():
            await self.settle()
        self.pending = None
        await self.send()
        await self.websocket.send_json({"type": "stopped"})


async def serve(websocket: WebSocket, transcriber: Transcriber) -> None:
    await websocket.accept()
    session: Session | None = None
    worker: asyncio.Task | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if session is not None and session.running:
                    session.push(message["bytes"])
                continue
            try:
                kind = json.loads(message.get("text") or "{}").get("type")
            except json.JSONDecodeError:
                continue
            if kind == "start":
                if worker is not None:  # a new recording replaces any unfinished one
                    session.running = False
                    worker.cancel()
                session = Session(websocket, transcriber)
                worker = asyncio.create_task(session.run())
            elif kind == "stop" and session is not None and worker is not None:
                session.running = False
                await worker
                worker = None
                try:
                    await session.finish()
                except Exception as error:  # the final read failed: report it
                    await websocket.send_json({"type": "error", "message": str(error)})
    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            session.running = False
        if worker is not None:
            worker.cancel()
