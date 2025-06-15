import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = [d["exact_match"] for d in data if "exact_match" in d]

    accuracy = sum(scores) / len(scores) if scores else 0.0

    plt.bar(["Exact Match"], [accuracy])
    plt.ylim(0, 1)
    plt.title("Accuracy (Exact Match)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_results("results/experiment_output.json")
