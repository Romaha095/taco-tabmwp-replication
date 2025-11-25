import os
import sys
from pathlib import Path
import requests


RAW_URLS = {
    "problems_train.json": "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/problems_train.json",
    "problems_dev.json": "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/problems_dev.json",
    "problems_test.json": "https://raw.githubusercontent.com/lupantech/PromptPG/main/data/tabmwp/problems_test.json",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target_path: Path) -> None:
    print(f"[*] Downloading {url} -> {target_path}")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[!] Failed to download {url}: {e}")
        sys.exit(1)

    target_path.write_bytes(resp.content)
    print(f"[+] Saved {target_path} ({len(resp.content)} bytes)")


def main() -> None:
    project_root = get_project_root()
    target_dir = project_root / "data" / "raw" / "tabmwp"
    ensure_dir(target_dir)

    print(f"[*] Project root: {project_root}")
    print(f"[*] Target directory for TABMWP: {target_dir}")

    for filename, url in RAW_URLS.items():
        target_path = target_dir / filename
        download_file(url, target_path)

    print("[✓] TABMWP JSON files downloaded successfully.")


if __name__ == "__main__":
    main()
