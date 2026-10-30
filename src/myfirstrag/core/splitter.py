"""
Paso 2 del pipeline — SPLIT (fragmentación).

Una página completa suele ser demasiado grande para recuperar información
precisa: el significado relevante se "diluye" en el resto del texto y además
gasta tokens del contexto del LLM. Por eso partimos cada `Document` en chunks.

`RecursiveCharacterTextSplitter` intenta cortar primero por párrafos ("\\n\\n"),
luego por líneas ("\\n"), luego por espacios... hasta que cada chunk mide como
máximo `chunk_size` caracteres. Es el splitter recomendado para texto genérico.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from myfirstrag.config import settings


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        # El solapamiento evita que una frase quede partida entre dos chunks
        # sin contexto en ninguno de ellos.
        chunk_overlap=settings.chunk_overlap,
        # Añade metadata["start_index"]: posición del chunk dentro de la página.
        add_start_index=True,
    )
    # Cada chunk hereda la metadata del Document original (source, page).
    return splitter.split_documents(docs)
