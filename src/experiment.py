import json
from pathlib import Path
from utils.data_loader import load_eval_dataset
from prompt_builder import PromptBuilder
from generator import AzureOpenAIGenerator
from parser import BasicParser
from evaluator import ExactMatchEvaluator
from dataclass import RAGData

def run_experiment(dataset_path: str, output_path: str):
    dataset = load_eval_dataset(dataset_path)

    builder = PromptBuilder()
    generator = AzureOpenAIGenerator()
    parser = BasicParser()
    evaluator = ExactMatchEvaluator()

    results = []

    for sample in dataset:
        sample = builder.forward(sample)
        sample = generator.forward(sample)
        sample = parser.forward(sample)
        sample = evaluator.forward(sample)
        results.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_experiment("data/eval_set.json", "results/experiment_output.json")
