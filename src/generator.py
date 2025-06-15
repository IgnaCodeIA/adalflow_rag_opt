import os
from dataclass import RAGData
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv('/Users/ignaciocarrenoromero/ProyectosPersonales/adalflow_rag_opt/.env')

class AzureOpenAIGenerator:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def forward(self, data: RAGData) -> RAGData:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data.prompt}
            ]
        )
        data.output = response.choices[0].message.content.strip()
        return data

if __name__ == "__main__":
    sample = RAGData(
        question="¿Quién fundó Microsoft?",
        documents=["Microsoft fue fundada por Bill Gates y Paul Allen."],
        prompt=(
            "Answer the question using only the information from the context below.\n\n"
            "Context:\nMicrosoft fue fundada por Bill Gates y Paul Allen.\n\n"
            "Question: ¿Quién fundó Microsoft?\nAnswer:"
        )
    )
    generator = AzureOpenAIGenerator()
    result = generator.forward(sample)
    print(result.output)