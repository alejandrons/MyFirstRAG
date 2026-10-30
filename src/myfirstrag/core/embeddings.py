"""
Paso 3 del pipeline — EMBED (vectorización).

Un modelo de embeddings convierte un texto en un vector de números (p. ej.
3072 dimensiones para `text-embedding-3-large`). Textos con significado
parecido producen vectores cercanos, lo que permite la "búsqueda semántica":
buscar por significado y no por coincidencia exacta de palabras.

`Embeddings` es una interfaz de LangChain con dos métodos:
    - embed_documents(list[str]) -> list[list[float]]   (al indexar)
    - embed_query(str)           -> list[float]         (al buscar)

Gracias a esa interfaz común, cambiar de proveedor (OpenAI, Google, Ollama...)
solo requiere cambiar esta función; el resto del proyecto no se entera.
"""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from myfirstrag.config import settings


def get_embeddings() -> Embeddings:
    # Lee OPENAI_API_KEY de las variables de entorno (cargadas en config.py).
    return OpenAIEmbeddings(model=settings.embedding_model)
