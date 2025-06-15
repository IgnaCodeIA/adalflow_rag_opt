import os
from adalflow import AdalFlow, run_optimization
from dataclass import RAGData
from prompt_builder import PromptBuilder
from generator import AzureOpenAIGenerator
from parser import BasicParser
from evaluator import ExactMatchEvaluator
from dataset import load_eval_dataset

def build_pipeline():
    return [
        PromptBuilder(),
        AzureOpenAIGenerator(),
        BasicParser(),
        ExactMatchEvaluator()
    ]

if __name__ == "__main__":
    dataset = load_eval_dataset("data/eval_set.json")

    flow = AdalFlow(
        steps=build_pipeline(),
        dataset=dataset,
        optimization_metric="exact_match",
        maximize=True,
    )

    run_optimization(
        flow=flow,
        search_space={
            "PromptBuilder.template": [
                "Contesta solo usando el contexto. Pregunta: {question}\nContexto:\n{documents}\nRespuesta:",
                "Usa únicamente la información del contexto para responder.\n\nContexto:\n{documents}\n\nPregunta: {question}\n\nRespuesta:",
                "Lee el contexto. Responde a la pregunta basada exclusivamente en esta información.\n\n{documents}\n\n{question}\n\nRespuesta:"
            ]
        },
        n_trials=10
    )
