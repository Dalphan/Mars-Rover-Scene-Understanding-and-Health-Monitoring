from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "REPO_VERSION.md"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    commit = git("rev-parse", "HEAD")
    short_commit = git("rev-parse", "--short", "HEAD")
    branch = git("branch", "--show-current") or "detached"
    commit_date = git("show", "-s", "--format=%cI", "HEAD")
    commit_subject = git("show", "-s", "--format=%s", "HEAD")
    commit_history = git("log", "--date=iso-strict", "--pretty=format:- `%h` `%cI` %s")
    dirty = bool(git("status", "--porcelain"))
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    VERSION_FILE.write_text(
        "\n".join(
            [
                "# Repository Version",
                "",
                "Use this file in Kaggle to verify which repository version was cloned.",
                "Regenerate it after a commit with `python scripts/update_repo_version.py`.",
                "",
                f"- Commit: `{commit}`",
                f"- Short commit: `{short_commit}`",
                f"- Branch: `{branch}`",
                f"- Commit date: `{commit_date}`",
                f"- Commit subject: {commit_subject}",
                f"- Working tree dirty when generated: `{dirty}`",
                f"- Generated at UTC: `{generated_at}`",
                "",
                "## Commit History",
                "",
                commit_history,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Updated {VERSION_FILE.relative_to(ROOT)} for {short_commit}")


if __name__ == "__main__":
    main()
