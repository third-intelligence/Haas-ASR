# Demos

Two browser demos that run on your own GPU, served by one local web server:

- **Long-form** (`/longform`): upload a recording of any length. It is cut into chunks
  of about 30 s, each cut placed at the quietest point near its position, and the
  chunks are transcribed in batches. The page shows the processing time, a waveform
  that fills in as chunks finish, and a timestamped transcript you can click to play
  from. The server reads the file with libsndfile, which accepts WAV, FLAC, MP3 and
  OGG (Vorbis or Opus) but not M4A, MP4 or WebM; convert those first, for example
  with `ffmpeg -i recording.m4a recording.flac`.
- **Real-time** (`/realtime`): speak into the microphone, or play an audio file, and
  see captions as you speak, with the end-to-end delay. Files are decoded by the
  browser, so any format it can play works.

> These demos are simplified implementations to show what the model can do, not for
> production use.

## Run

From a clone of this repository, on a Linux machine with an NVIDIA GPU:

```bash
uv sync --group demo
uv run python -m demos.app
```

Then open <http://localhost:7860>. The model is downloaded from Hugging Face on first
run; pass `--model path/to/model.nemo` to use a local file. Other options:
`--port`, `--device`, `--max-upload-mb`, and `--host` (see below).

The server listens on `127.0.0.1` only. If the GPU machine is remote, forward the port
over SSH:

```bash
ssh -L 7860:127.0.0.1:7860 <gpu-machine>
```

`--host 0.0.0.0` makes the server reachable from other machines, but it has no
authentication: use it only on a trusted network. Browsers allow microphone access
only on `localhost` or over HTTPS, so open the page through `localhost` (for example
with the SSH forward above).

## How the real-time demo works

Haas-ja-v1.0 is an offline model: it reads a whole utterance at once and was not
trained for streaming. The demo makes it feel live by re-reading the phrase being
spoken about every 250 ms, which is fast enough because each read takes tens of
milliseconds.

- **Pending** text (light green) is the latest read of the current phrase. It can
  change as more audio arrives.
- A phrase is **settled** (dark) after 0.7 s of silence, or when it reaches 14 s, in
  which case it is cut at its quietest point and the rest starts the next phrase.
  The settled phrase is read once more in full and then never changes.
- The **delay** is measured from the newest audio leaving the browser's audio thread
  to the updated text appearing on screen, including the network, on the page's own
  clock. The page shows the median of the last 20 updates.

Speech and silence are told apart by a fixed volume threshold (about -38 dBFS). With
steady background noise above it, pauses are not detected, so phrases settle only at
the 14 s limit; with a very quiet microphone, speech can be missed. The segmentation
logic is in [`segmenter.py`](segmenter.py) and its thresholds, including
`silence_rms`, in `SegmenterConfig`.

## Files

| file | what it does |
|---|---|
| `app.py` | the web server and command line |
| `inference.py` | loads the model and runs every call on one dedicated thread |
| `longform.py` | long-form WebSocket: receive a file, chunk, transcribe in batches |
| `realtime.py` | real-time WebSocket: receive audio frames, re-read and settle phrases |
| `segmenter.py` | splits the incoming audio into phrases (tested in `tests/test_demo_segmenter.py`) |
| `static/` | the two pages, their scripts, the shared stylesheet and the audio-capture worklet |
