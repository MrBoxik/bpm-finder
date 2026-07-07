from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from audio_formats import is_audio_file
from bpm_finder import (
    DEFAULT_DEEP_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_ANALYZE_SECONDS,
    BpmDetectionError,
    estimate_bpm,
    format_confidence_threshold,
    parse_confidence_threshold,
)
from exporters import write_excel_csv, write_xlsx


def _default_worker_count() -> int:
    logical_cpus = os.cpu_count() or 1
    if logical_cpus <= 2:
        return logical_cpus
    return max(2, min(12, round(logical_cpus * 0.75)))


DEFAULT_WORKERS = _default_worker_count()
HEADERS = [
    "file",
    "bpm",
    "confidence",
    "confidence_percent",
    "status",
    "error",
    "duration_seconds",
    "analyzed_seconds",
    "path",
]


def analyze_folder(
    folder: Path,
    output: Path,
    *,
    workers: int,
    max_seconds: float,
    deep_confidence_threshold: float,
    limit: int | None = None,
    xlsx_output: Path | None = None,
) -> int:
    paths = sorted(
        (path for path in folder.rglob("*") if is_audio_file(path)),
        key=lambda path: str(path).casefold(),
    )
    if limit is not None:
        paths = paths[:limit]

    started = time.perf_counter()
    finished = 0
    errors = 0
    rows: list[list[str | float]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_one, path, max_seconds, deep_confidence_threshold): path
            for path in paths
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            finished += 1
            if row[4] != "ok":
                errors += 1

            if finished == 1 or finished % 100 == 0 or finished == len(paths):
                elapsed = max(0.01, time.perf_counter() - started)
                rate = finished / elapsed
                print(
                    f"{finished}/{len(paths)} done "
                    f"({rate:.1f} files/sec, errors={errors})",
                    flush=True,
                )

    rows.sort(key=lambda row: str(row[-1]).casefold())
    write_excel_csv(output, HEADERS, rows)
    if xlsx_output is not None:
        write_xlsx(xlsx_output, HEADERS, rows)

    elapsed = max(0.01, time.perf_counter() - started)
    print(
        f"Finished {finished} file(s) in {elapsed:.1f}s "
        f"({finished / elapsed:.1f} files/sec). Output: {output}",
        flush=True,
    )
    return 0 if errors == 0 else 1


def _analyze_one(
    path: Path,
    max_seconds: float,
    deep_confidence_threshold: float,
) -> list[str | float]:
    try:
        result = estimate_bpm(
            path,
            max_analyze_seconds=max_seconds,
            deep_confidence_threshold=deep_confidence_threshold,
        )
        return [
            path.name,
            result.bpm,
            result.confidence,
            int(max(0.0, min(1.0, result.confidence)) * 100),
            "ok",
            "",
            result.duration_seconds,
            result.analyzed_seconds,
            str(path),
        ]
    except BpmDetectionError as exc:
        return [path.name, "", "", "", "error", str(exc), "", "", str(path)]
    except Exception as exc:
        return [path.name, "", "", "", "error", f"Unexpected error: {exc}", "", "", str(path)]


def _main() -> int:
    parser = argparse.ArgumentParser(description="Find BPM for a folder of audio files to CSV.")
    parser.add_argument("folder", help="Folder to scan recursively for audio files.")
    parser.add_argument("--output", default="bpm_results.csv", help="CSV output path.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_ANALYZE_SECONDS)
    parser.add_argument(
        "--deep-confidence",
        default=format_confidence_threshold(DEFAULT_DEEP_CONFIDENCE_THRESHOLD),
        help=(
            "Run deeper analysis when confidence is this value or lower. "
            "Accepts values like 99, 99%%, or 0.99."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional test limit.")
    parser.add_argument("--xlsx-output", default=None, help="Optional Excel .xlsx output path.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder does not exist: {folder}", flush=True)
        return 2

    max_workers = max(1, min(24, os.cpu_count() or 1))
    workers = max(1, min(max_workers, args.workers))
    try:
        deep_confidence_threshold = parse_confidence_threshold(args.deep_confidence)
    except ValueError as exc:
        print(str(exc), flush=True)
        return 2

    return analyze_folder(
        folder,
        Path(args.output),
        workers=workers,
        max_seconds=max(20.0, min(300.0, args.max_seconds)),
        deep_confidence_threshold=deep_confidence_threshold,
        limit=args.limit,
        xlsx_output=Path(args.xlsx_output) if args.xlsx_output else None,
    )


if __name__ == "__main__":
    raise SystemExit(_main())
