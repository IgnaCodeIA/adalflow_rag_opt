# Adalflow_rag_opt

Este proyecto es una herramienta de evaluación para sistemas RAG que permite probar la calidad de las respuestas generadas por un modelo de lenguaje a partir de un contexto documental usando Azure OpenAI

## Qué hace esta herramienta

Lanza un pipeline que toma una pregunta y una serie de documentos de contexto, construye un prompt, lo envía al modelo desplegado en Azure OpenAI y evalúa la respuesta generada con métricas como exact match

## Requisitos previos

- Python 3.10 o superior  
- Acceso a una cuenta de Azure con Azure OpenAI habilitado  
- Archivo `.env` con las variables necesarias

## Variables necesarias en el archivo .env

AZURE_OPENAI_API_KEY  
AZURE_OPENAI_ENDPOINT  
AZURE_OPENAI_API_VERSION  
AZURE_OPENAI_DEPLOYMENT_NAME  

## Instalación y ejecución

1 descomprime el proyecto en tu entorno de trabajo  
2 entra al directorio del proyecto  
3 instala las dependencias con

```bash
pip install -r requirements.txt
```
4 asegúrate de tener el archivo .env correctamente configurado
5 ejecuta el pipeline con

```bash
python src/run_pipeline.py
```

## Estructura del proyecto
	•	src/dataclass.py contiene la estructura de datos RAGData
	•	src/generator.py implementa la generación con Azure OpenAI
	•	src/evaluator.py evalúa la respuesta generada
	•	src/run_pipeline.py ejecuta el flujo completo de prueba
