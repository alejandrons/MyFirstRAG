"""
Configuración centralizada del proyecto.

Todas las capas (core, database, services, ui) leen sus parámetros desde aquí,
así no hay "números mágicos" repartidos por el código y es fácil experimentar
cambiando un solo valor (o una variable de entorno en el archivo `.env`).
"""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import find_dotenv, load_dotenv

# Carga las variables del archivo .env (OPENAI_API_KEY, etc.) en os.environ.
# Los clientes de LangChain (OpenAIEmbeddings, ChatOpenAI...) leen la API key
# directamente de las variables de entorno, por eso basta con cargarlas aquí.
load_dotenv(find_dotenv())

# Raíz del repositorio: src/myfirstrag/config.py -> sube 3 niveles.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # --- Almacenamiento ---------------------------------------------------
    # Carpeta donde se guardan los archivos subidos desde la UI.
    docs_dir: Path = PROJECT_ROOT / "docs"
    # Carpeta donde ChromaDB persiste la base de datos vectorial en disco.
    chroma_dir: Path = PROJECT_ROOT / "chroma_langchain_db"
    # Una "colección" en Chroma es equivalente a una tabla en SQL.
    collection_name: str = os.getenv("CHROMA_COLLECTION", "myfirstrag")

    # --- Indexación -------------------------------------------------------
    # Tamaño máximo (en caracteres) de cada fragmento (chunk).
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    # Caracteres compartidos entre chunks consecutivos para no cortar ideas.
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Modelos ----------------------------------------------------------
    # Modelo que convierte texto en vectores (embeddings).
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    # Modelo de chat en formato "proveedor:modelo" (lo entiende init_chat_model).
    chat_model: str = os.getenv("CHAT_MODEL", "openai:gpt-5-mini")

    # --- Recuperación -----------------------------------------------------
    # Cuántos chunks devuelve la búsqueda por similitud.
    top_k: int = int(os.getenv("TOP_K", "4"))


settings = Settings()

# Modelos de chat que se ofrecen en el selector de la UI ("proveedor:modelo").
# Cada proveedor necesita su paquete de integración (langchain-openai,
# langchain-anthropic, langchain-google-genai) y su API key en `.env`.
# Desde la UI también se puede escribir cualquier otro modelo a mano.
CHAT_MODEL_OPTIONS: list[str] = list(
    dict.fromkeys(
        [
            settings.chat_model,
            "openai:gpt-5-mini",
            "openai:gpt-5",
            "anthropic:claude-haiku-5-5",
            "anthropic:claude-sonnet-5-5",
            "anthropic:claude-opus-5-5",
            "google_genai:gemini-2.5-flash",
            "google_genai:gemini-2.5-pro",
        ]
    )
)
