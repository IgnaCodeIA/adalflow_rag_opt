import json
from pathlib import Path
from typing import List
from dataclass import RAGData

def load_eval_dataset(path: str) -> List[RAGData]:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [RAGData(**item) for item in raw_data]

if __name__ == "__main__":
    dataset = load_eval_dataset("data/eval_set.json")
    for d in dataset:
        print(d)
