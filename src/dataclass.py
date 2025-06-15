# Aquí se definirá el DataClass (RAGData)
from dataclasses import dataclass, field
from typing import List, Optional
from adalflow import DataClass

@dataclass
class RAGData(DataClass):
    question: str
    documents: List[str] = field(default_factory=list)
    prompt: Optional[str] = None
    output: Optional[str] = None
    expected_answer: Optional[str] = None
    exact_match: Optional[float] = None  

    __input_fields__ = ["question", "documents"]
    __output_fields__ = ["output", "exact_match"]

if __name__ == "__main__":
    sample = RAGData(
        question="¿Quién fundó Microsoft?",
        documents=["Microsoft fue fundada por Bill Gates y Paul Allen."],
        expected_answer="Bill Gates"
    )

    print("DataClass creado con éxito:")
    print(sample)
    print("Exportado como dict:")
    print(sample.to_dict())
    print("Exportado como JSON:")
    print(sample.to_json())
