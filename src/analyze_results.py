import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze(results):
    df = pd.DataFrame(results)
    df["correct"] = df["exact_match"] == 1.0

    accuracy = df["correct"].mean()
    print(f" Accuracy total: {accuracy:.2%}")
    print(f" Total ejemplos: {len(df)}")
    print(f" Fallos: {len(df) - df['correct'].sum()}")

    print("\n📉 Ejemplos fallidos:")
    print(df[~df["correct"]][["question", "expected_answer", "generated_answer"]])

    df["exact_match"].plot(kind="hist", bins=10, title="Distribución de Exact Match")
    plt.xlabel("Exact Match Score")
    plt.ylabel("Número de ejemplos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    result_path = Path("data/results.json")
    results = load_results(result_path)
    analyze(results)
