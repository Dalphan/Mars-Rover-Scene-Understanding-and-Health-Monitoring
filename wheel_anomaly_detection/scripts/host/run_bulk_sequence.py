"""Run clean/anomaly bulk steps sequentially with deterministic resume."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--blender-executable", type=Path, required=True)
    args = parser.parse_args()
    sequence_path = args.sequence.resolve()
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if int(sequence.get("schema_version", 0)) != 1 or not sequence.get("steps"):
        raise ValueError("Invalid bulk sequence config")
    status_path = REPO_ROOT / sequence["status_path"]
    status = {
        "schema_version": 1,
        "sequence_id": sequence["sequence_id"],
        "status": "running",
        "started_at": _utc_now(),
        "steps": [],
    }
    _atomic_json(status_path, status)
    try:
        for position, step in enumerate(sequence["steps"]):
            runner = step["runner"]
            if runner not in {"clean", "anomaly"}:
                raise ValueError(f"Unknown runner: {runner}")
            config_path = REPO_ROOT / step["config"]
            config = json.loads(config_path.read_text(encoding="utf-8"))
            run_dir = REPO_ROOT / config["output"]["root"] / step["run_id"]
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "host" / ("run_clean_batch.py" if runner == "clean" else "run_anomaly_batch.py")),
                "--config", str(config_path),
                "--run-id", step["run_id"],
                "--blender-executable", str(args.blender_executable.resolve()),
            ]
            if run_dir.exists():
                command.append("--resume")
            step_status = {"position": position, **step, "status": "running", "started_at": _utc_now(), "run_dir": run_dir.relative_to(REPO_ROOT).as_posix()}
            status["steps"].append(step_status)
            _atomic_json(status_path, status)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            step_status["status"] = "complete"
            step_status["finished_at"] = _utc_now()
            _atomic_json(status_path, status)
        status["status"] = "complete"
        status["finished_at"] = _utc_now()
        _atomic_json(status_path, status)
    except Exception as error:
        status["status"] = "failed"
        status["failed_at"] = _utc_now()
        status["error"] = f"{type(error).__name__}: {error}"
        _atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
