"""Local web server for the Haas demos.

    uv sync --group demo
    uv run python -m demos.app --model third-intelligence/Haas-ja-v1.0

then open http://localhost:7860. Two pages share one model:

    /longform   upload a recording of any length and watch it being transcribed
    /realtime   speak into the microphone, or play a file, and see live captions

The server listens on 127.0.0.1 only. Use --host 0.0.0.0 to reach it from other
machines; it has no authentication, so do that only on a trusted network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from . import longform, realtime
from .inference import Transcriber

STATIC = Path(__file__).parent / "static"


def create_app(transcriber: Transcriber, model_name: str, max_upload_mb: int) -> FastAPI:
    app = FastAPI(title="Haas demos")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def revalidate(request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Browsers otherwise reuse cached pages and scripts after the demo is updated.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def home() -> RedirectResponse:
        return RedirectResponse("/longform")

    @app.get("/longform")
    def longform_page() -> FileResponse:
        return FileResponse(STATIC / "longform.html")

    @app.get("/realtime")
    def realtime_page() -> FileResponse:
        return FileResponse(STATIC / "realtime.html")

    @app.get("/api/info")
    def info() -> dict:
        return {"model": model_name}

    @app.websocket("/ws/longform")
    async def longform_socket(websocket: WebSocket) -> None:
        await longform.serve(websocket, transcriber, max_upload_mb * 1_000_000)

    @app.websocket("/ws/realtime")
    async def realtime_socket(websocket: WebSocket) -> None:
        await realtime.serve(websocket, transcriber)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Haas long-form and real-time transcription demos")
    parser.add_argument(
        "--model",
        default="third-intelligence/Haas-ja-v1.0",
        help="Hugging Face model id or .nemo path",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-upload-mb", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    transcriber = Transcriber(args.model, device=args.device)
    transcriber.warm_up()
    print(f"Ready on {transcriber.device_name}: http://{args.host}:{args.port}")
    app = create_app(transcriber, Path(args.model).name, args.max_upload_mb)
    uvicorn.run(app, host=args.host, port=args.port, http="h11", ws="wsproto", log_level="warning")


if __name__ == "__main__":
    main()
