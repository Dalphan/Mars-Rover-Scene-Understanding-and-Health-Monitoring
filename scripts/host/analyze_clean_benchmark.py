"""Compare deterministic clean-render benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def _report(run_dir: Path) -> dict:
    reports = list((run_dir / "_staging").glob("chunk_*/chunk_report.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one chunk report in {run_dir}")
    return json.loads(reports[0].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline_64_fresh")
    parser.add_argument("--cache", default="cache_64")
    parser.add_argument("--pipeline", default="cache_64_pipeline_bounded")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = [args.baseline, args.cache, args.pipeline]
    runs = {}
    for name in names:
        run_dir = args.root / name
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        rows = _rows(run_dir)
        report = _report(run_dir)
        runs[name] = {
            "sample_count": len(rows),
            "wall_seconds": float(run["performance"]["wall_seconds"]),
            "pipeline_seconds": float(report["chunk_pipeline_seconds"]),
            "render_seconds": sum(float(row["render"]["render_seconds"]) for row in rows),
            "cpu_postprocess_seconds": float(report["cpu_postprocess_seconds"]),
            "settings": report["benchmark"],
        }
    baseline_rows = {row["sample_id"]: row for row in _rows(args.root / args.baseline)}
    comparisons = []
    baseline_seconds = runs[args.baseline]["pipeline_seconds"]
    for name in names[1:]:
        candidate_rows = {row["sample_id"]: row for row in _rows(args.root / name)}
        rgb_equal = sum(
            baseline_rows[sample_id]["artifacts"]["rgb"]["sha256"]
            == candidate_rows[sample_id]["artifacts"]["rgb"]["sha256"]
            for sample_id in baseline_rows
        )
        mask_equal = sum(
            baseline_rows[sample_id]["artifacts"]["target_wheel_mask"]["sha256"]
            == candidate_rows[sample_id]["artifacts"]["target_wheel_mask"]["sha256"]
            for sample_id in baseline_rows
        )
        semantic_equal = all(
            baseline_rows[sample_id]["sampling"] == candidate_rows[sample_id]["sampling"]
            and baseline_rows[sample_id]["seeds"] == candidate_rows[sample_id]["seeds"]
            and all(
                baseline_rows[sample_id]["resolved"][key] == candidate_rows[sample_id]["resolved"][key]
                for key in ("camera_matrix_world", "wheel_matrix_world", "rover_matrix_world")
            )
            for sample_id in baseline_rows
        )
        seconds = runs[name]["pipeline_seconds"]
        comparisons.append(
            {
                "candidate": name,
                "saved_seconds": baseline_seconds - seconds,
                "saved_fraction": 1.0 - seconds / baseline_seconds,
                "speedup": baseline_seconds / seconds,
                "rgb_byte_equal": {"equal": rgb_equal, "total": len(baseline_rows)},
                "mask_byte_equal": {"equal": mask_equal, "total": len(baseline_rows)},
                "semantic_lock_equal": semantic_equal,
            }
        )
    output = {"schema_version": 1, "runs": runs, "comparisons": comparisons}
    target = args.output or args.root / "benchmark_report.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
