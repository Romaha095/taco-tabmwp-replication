import json
import os

RAW_DIR = "data/raw/tabmwp"

def load_file(name):
    path = os.path.join(RAW_DIR, name)
    return json.load(open(path, "r", encoding="utf-8"))

if __name__ == "__main__":
    train = load_file("problems_train.json")
    print("Train size:", len(train))
    ex = list(train.values())[0]
    print(json.dumps(ex, indent=2, ensure_ascii=False))
