from dataclass import RAGData

class BasicParser:
    def forward(self, data: RAGData) -> RAGData:
        if data.output:
            data.output = data.output.strip()
        return data

if __name__ == "__main__":
    sample = RAGData(
        question="¿Quién fundó Microsoft?",
        documents=["Microsoft fue fundada por Bill Gates y Paul Allen."],
        prompt="Answer the question using only the information from the context below.\n\nContext:\nMicrosoft fue fundada por Bill Gates y Paul Allen.\n\nQuestion: ¿Quién fundó Microsoft?\nAnswer:",
        output="  Bill Gates "
    )

    parser = BasicParser()
    parsed = parser.forward(sample)
    print(f"Parsed output: '{parsed.output}'")
