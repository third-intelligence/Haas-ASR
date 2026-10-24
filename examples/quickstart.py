"""Transcribe short audio files (up to about 30 s each; any sample rate, mono or stereo).

python examples/quickstart.py --model <hf-id-or-.nemo> example1.wav example2.wav
"""

import argparse

import haas_asr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face model id or .nemo path")
    parser.add_argument("audio", nargs="+", help="audio files")
    args = parser.parse_args()

    model = haas_asr.load_model(args.model)
    audio = [haas_asr.load_audio(path) for path in args.audio]
    for path, hypothesis in zip(args.audio, model.transcribe(audio), strict=True):
        print(f"{path}\t{hypothesis.text.strip()}")


if __name__ == "__main__":
    main()
