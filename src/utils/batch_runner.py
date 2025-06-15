from utils.data_loader import load_eval_dataset
from prompt_builder import PromptBuilder
from generator import AzureOpenAIGenerator
from parser import BasicParser
from evaluator import ExactMatchEvaluator
import json

def run_batch(dataset_path: str, output_path: str):
    dataset = load_eval_dataset(dataset_path)
    
    builder = PromptBuilder()
    generator = AzureOpenAIGenerator()
    parser = BasicParser()
    evaluator = ExactMatchEvaluator()

    results = []

    for i, sample in enumerate(dataset):
        sample = builder.forward(sample)
        sample = generator.forward(sample)
        sample = parser.forward(sample)
        sample = evaluator.forward(sample)

        results.append({
            "question": sample.question,
            "documents": sample.documents,
            "expected_answer": sample.expected_answer,
            "generated_answer": sample.output,
            "prompt": sample.prompt,
            "exact_match": sample.exact_match
        })
        print(f"✅ Procesado ejemplo {i + 1}/{len(dataset)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_batch("data/eval_set.json", "data/results.json")
