"""Transcribe one audio file of any length.

Audio longer than 30 s is cut at quiet points and the chunks are transcribed in
batches; shorter audio is transcribed in one piece.

python examples/transcribe_longform.py --model <hf-id-or-.nemo> lecture.flac
"""

import argparse

import haas_asr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face model id or .nemo path")
    parser.add_argument("audio", help="audio file")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = haas_asr.load_model(args.model)
    audio = haas_asr.load_audio(args.audio)
    print(haas_asr.transcribe_longform(model, audio, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
