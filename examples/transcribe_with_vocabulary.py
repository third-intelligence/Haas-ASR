"""Transcribe one audio file of any length with a custom vocabulary.

Each --words entry is a word as it should be written, optionally followed by "=" and
its readings, separated by commas: the spellings Haas may write for it.

python examples/transcribe_with_vocabulary.py --model <hf-id-or-.nemo> example.flac \\
    --words "Haas=ハース" "Third Intelligence=サードインテリジェンス"
"""

import argparse

import haas_asr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face model id or .nemo path")
    parser.add_argument("audio", help="audio file")
    parser.add_argument("--words", nargs="+", required=True, help='"written form=reading,..."')
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    vocabulary = haas_asr.Vocabulary.parse(args.words)
    model = haas_asr.load_model(args.model)
    audio = haas_asr.load_audio(args.audio)
    print(
        haas_asr.transcribe_longform(
            model, audio, batch_size=args.batch_size, vocabulary=vocabulary
        )
    )


if __name__ == "__main__":
    main()
