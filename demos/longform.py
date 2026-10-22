"""Long-form demo: upload a recording, transcribe it in chunks, stream the results.

Protocol on the WebSocket, one recording per message sequence:

    page -> server  {"type": "start", "bytes": N, "batch_size": B}, then the file in binary frames
    server -> page  {"type": "progress", "received": n, "total": N}         while receiving
                    {"type": "plan", "duration", "chunks", "peaks"}          once decoded
                    {"type": "chunks", "items": [...], "done"}               per batch
                    {"type": "done", "duration", "chunks"}
                    {"type": "error", "message"}

The page times the run itself, from the plan to "done": chunking and
transcription, not the upload or decoding of the file.
"""

from __future__ import annotations

import asyncio
import io
import json
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

import haas_asr
from haas_asr.audio import SAMPLE_RATE

from .inference import Transcriber

PEAK_BINS = 1600


def waveform_peaks(audio: np.ndarray, bins: int = PEAK_BINS) -> list[int]:
    """Max |amplitude| per bin, scaled to 0-255, for drawing the waveform."""
    if audio.size == 0:
        return []
    edges = np.linspace(0, audio.size, min(bins, audio.size) + 1, dtype=np.int64)
    peaks = np.maximum.reduceat(np.abs(audio), edges[:-1])
    top = float(np.percentile(peaks, 99.5)) or 1.0
    return np.clip(peaks / top * 255, 0, 255).astype(int).tolist()


async def receive_file(websocket: WebSocket, total: int) -> bytes:
    """Read `total` bytes of binary frames, reporting progress about every 200 ms."""
    data = bytearray()
    last_report = 0.0
    while len(data) < total:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()
        if message.get("bytes"):
            data += message["bytes"]
            now = time.perf_counter()
            if now - last_report > 0.2 or len(data) >= total:
                last_report = now
                await websocket.send_json(
                    {"type": "progress", "received": len(data), "total": total}
                )
    return bytes(data)


async def transcribe_recording(
    websocket: WebSocket, transcriber: Transcriber, raw: bytes, batch_size: int
) -> None:
    audio = await asyncio.to_thread(haas_asr.load_audio, io.BytesIO(raw))
    bounds = haas_asr.split_at_quiet_points(audio)
    duration = audio.size / SAMPLE_RATE
    await websocket.send_json(
        {
            "type": "plan",
            "duration": duration,
            "chunks": [[start / SAMPLE_RATE, end / SAMPLE_RATE] for start, end in bounds],
            "peaks": waveform_peaks(audio),
        }
    )
    for first in range(0, len(bounds), batch_size):
        group = bounds[first : first + batch_size]
        texts, _ = await transcriber.transcribe(
            [audio[start:end] for start, end in group], batch_size=len(group)
        )
        await websocket.send_json(
            {
                "type": "chunks",
                "items": [
                    {
                        "index": first + i,
                        "start": s / SAMPLE_RATE,
                        "end": e / SAMPLE_RATE,
                        "text": t,
                    }
                    for i, ((s, e), t) in enumerate(zip(group, texts, strict=True))
                ],
                "done": first + len(group),
            }
        )
    await websocket.send_json({"type": "done", "duration": duration, "chunks": len(bounds)})


async def serve(websocket: WebSocket, transcriber: Transcriber, max_upload_bytes: int) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("text") is None:
                continue  # the rest of a refused or abandoned upload
            try:
                request = json.loads(message["text"])
            except json.JSONDecodeError:
                continue
            if request.get("type") != "start":
                continue
            total = int(request.get("bytes") or 0)
            batch_size = max(1, int(request.get("batch_size") or 64))
            if not 0 < total <= max_upload_bytes:
                limit = max_upload_bytes // 1_000_000
                await websocket.send_json(
                    {"type": "error", "message": f"Files up to {limit} MB are accepted."}
                )
                continue
            raw = await receive_file(websocket, total)
            try:
                await transcribe_recording(websocket, transcriber, raw, batch_size)
            except Exception as error:  # report decode and inference errors to the page
                await websocket.send_json({"type": "error", "message": str(error)})
    except WebSocketDisconnect:
        pass
