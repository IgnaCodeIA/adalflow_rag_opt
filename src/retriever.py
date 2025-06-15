# Simulación de componente Retriever
from adalflow import Component
from dataclass import RAGData

class StaticRetriever(Component):
    def forward(self, data: RAGData) -> RAGData:
        return data

if __name__ == "__main__":
    sample = RAGData(
        question="What is the closest planet to the Sun?",
        documents=[
            "Mercury is the closest planet to the Sun.",
            "Venus is the second planet from the Sun."
        ]
    )

    retriever = StaticRetriever()
    result = retriever.forward(sample)

    print("Retriever output:")
    print(result.documents)
