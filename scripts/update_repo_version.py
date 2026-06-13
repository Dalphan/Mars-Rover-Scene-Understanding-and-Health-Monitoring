from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "REPO_VERSION.txt"


def main() -> None:
    current = 0
    if VERSION_FILE.exists():
        content = VERSION_FILE.read_text(encoding="utf-8").strip()
        current = int(content) if content else 0

    new_version = current + 1
    VERSION_FILE.write_text(f"{new_version}\n", encoding="utf-8")
    print(f"Updated {VERSION_FILE.name} to {new_version}")


if __name__ == "__main__":
    main()
