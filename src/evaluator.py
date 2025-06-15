from dataclass import RAGData

class ExactMatchEvaluator:
    def forward(self, data: RAGData) -> RAGData:
        if data.output is None or data.expected_answer is None:
            data.exact_match = 0.0
        else:
            data.exact_match = float(data.output.strip().lower() == data.expected_answer.strip().lower())
        return data

if __name__ == "__main__":
    sample = RAGData(
        question="¿Quién fundó Microsoft?",
        documents=["Microsoft fue fundada por Bill Gates y Paul Allen."],
        output="Bill Gates",
        expected_answer="Bill Gates"
    )

    evaluator = ExactMatchEvaluator()
    evaluated = evaluator.forward(sample)
    print("Exact Match:", evaluated.exact_match)
