"""Evaluate a model on asr-ja-bench, the benchmark the model card reports.

asr-ja-bench ships no audio or transcripts; build it first with the rebuild.py
in https://huggingface.co/datasets/third-intelligence/asr-ja-bench, then point
--bench at the benchmark data directory it wrote (its --out), not at the
downloaded dataset repository:

uv sync --group eval
uv run python examples/evaluate_asr_ja_bench.py \\
    --model third-intelligence/Haas-ja-v1.0 --bench path/to/asr-ja-bench-v1.1

The settings match the published evaluation: bfloat16, batched greedy
decoding, batches of 512 in the benchmark's order. Transcripts are scored with
jacer (CER strict, flex and kana). CER is micro-averaged within each subset,
and the overall score weights the subsets by audio duration.
"""

import argparse
import json
from pathlib import Path

import jacer
import numpy as np
import soundfile as sf

import haas_asr

SUBSETS = ("jsut", "jvnv", "tedxjp10k", "fleurs", "speech-bsd")
DATASET_URL = "https://huggingface.co/datasets/third-intelligence/asr-ja-bench"


def check_bench(bench: Path, subsets: tuple[str, ...]) -> None:
    """Stop with a clear message unless `bench` is rebuild.py's output."""
    if (bench / "rebuild.py").is_file() or (bench / "data").is_dir():
        raise SystemExit(
            f"{bench} is the downloaded dataset repository, not the benchmark data. "
            "Run its rebuild.py and pass the directory given as --out."
        )
    missing = [s for s in subsets if not (bench / f"{s}.jsonl").is_file()]
    if missing:
        raise SystemExit(
            f"{bench} has no {', '.join(f'{s}.jsonl' for s in missing)}. --bench must be the "
            f"benchmark data directory written by rebuild.py from {DATASET_URL}"
        )


def read_subset(bench: Path, subset: str) -> list[dict]:
    with open(bench / f"{subset}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def transcribe(model, bench: Path, rows: list[dict], batch_size: int) -> list[str]:
    hypotheses = []
    for i in range(0, len(rows), batch_size):
        audio = []
        for r in rows[i : i + batch_size]:
            x, sr = sf.read(bench / r["audio"], dtype="float32")
            if sr != haas_asr.SAMPLE_RATE:
                raise SystemExit(f"{r['audio']}: {sr} Hz, expected {haas_asr.SAMPLE_RATE}")
            audio.append(np.ascontiguousarray(x))
        out = model.transcribe(audio, batch_size=len(audio), verbose=False)
        hypotheses += [h.text.strip() for h in out]
    return hypotheses


def score_line(name: str, samples: int, cer: dict[str, float]) -> str:
    return f"{name:11s} {samples:6d} samples  " + "  ".join(
        f"{m} {cer[m] * 100:5.2f}" for m in jacer.METRICS
    )


def write_hypotheses(path: Path, rows: list[dict], hypotheses: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r, h in zip(rows, hypotheses, strict=True):
            record = {"id": r["id"], "reference": r["text"], "hypothesis": h}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face model id or .nemo path")
    parser.add_argument(
        "--bench",
        type=Path,
        required=True,
        help="benchmark data directory written by asr-ja-bench's rebuild.py (its --out)",
    )
    parser.add_argument(
        "--subset", action="append", choices=SUBSETS, help="repeatable; default: all five"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--out",
        type=Path,
        help="also write per-sample hypotheses (<subset>.jsonl) and summary.json here",
    )
    args = parser.parse_args()

    if jacer.METRICS_VERSION != "v1.2":
        print(f"note: jacer metrics {jacer.METRICS_VERSION}; the model card uses v1.2")
    subsets = tuple(args.subset or SUBSETS)
    check_bench(args.bench, subsets)
    model = haas_asr.load_model(args.model)

    results = {}
    for subset in subsets:
        rows = read_subset(args.bench, subset)
        hyps = transcribe(model, args.bench, rows, args.batch_size)
        cer = jacer.cer([r["text"] for r in rows], hyps, jacer.METRICS)
        hours = sum(r["duration"] for r in rows) / 3600
        results[subset] = {"samples": len(rows), "hours": hours, **cer}
        print(score_line(subset, len(rows), cer), flush=True)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            write_hypotheses(args.out / f"{subset}.jsonl", rows, hyps)

    total = sum(r["hours"] for r in results.values())
    weighted = {m: sum(r[m] * r["hours"] for r in results.values()) / total for m in jacer.METRICS}
    print(score_line("weighted", sum(r["samples"] for r in results.values()), weighted))
    if args.out:
        summary = {
            "model": str(args.model),
            "subsets": results,
            "weighted": weighted,
            "metrics_version": jacer.METRICS_VERSION,
            "scoring": jacer.scoring_environment(),
        }
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
