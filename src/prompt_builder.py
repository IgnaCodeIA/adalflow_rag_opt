from dataclass import RAGData

class PromptBuilder:
    def __init__(self, template=None):
        self.template = template or (
            "Answer the question using only the information from the context below.\n\n"
            "Context:\n{documents}\n\nQuestion: {question}\nAnswer:"
        )

    def forward(self, data: RAGData) -> RAGData:
        prompt = self.template.format(
            documents="\n".join(data.documents),
            question=data.question
        )
        data.prompt = prompt
        return data

if __name__ == "__main__":
    sample = RAGData(
        question="¿Quién fundó Microsoft?",
        documents=["Microsoft fue fundada por Bill Gates y Paul Allen."]
    )
    builder = PromptBuilder()
    result = builder.forward(sample)
    print(result.prompt)
