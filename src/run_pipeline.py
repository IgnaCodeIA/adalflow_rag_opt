# Script principal para correr pipelinesfrom dataclass import RAGData
from prompt_builder import PromptBuilder
from generator import AzureOpenAIGenerator
from parser import BasicParser
from evaluator import ExactMatchEvaluator
from dataclass import RAGData

def run_sample_pipeline():
    sample = RAGData(
        question="¿Quién fundó Microsoft?",
        documents=["Microsoft fue fundada por Bill Gates y Paul Allen."],
        expected_answer="Bill Gates"
    )

    builder = PromptBuilder()
    generator = AzureOpenAIGenerator()
    parser = BasicParser()
    evaluator = ExactMatchEvaluator()

    sample = builder.forward(sample)
    sample = generator.forward(sample)
    sample = parser.forward(sample)
    sample = evaluator.forward(sample)

    print("Pregunta:", sample.question)
    print("Prompt generado:", sample.prompt)
    print("Respuesta del modelo:", sample.output)
    print("Respuesta esperada:", sample.expected_answer)
    print("Puntuación Exact Match:", sample.exact_match)

if __name__ == "__main__":
    run_sample_pipeline()
