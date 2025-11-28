import json
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


project_root = get_project_root()
RAW_DIR = project_root / "data" / "raw" / "tabmwp"


def load_file(name: str):
    path = RAW_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    train = load_file("problems_train.json")
    print("Train size:", len(train))

    ex = list(train.values())[0]
    print(json.dumps(ex, indent=2, ensure_ascii=False))
