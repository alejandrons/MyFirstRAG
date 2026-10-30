"""
Paso 4 del pipeline — STORE (almacenamiento vectorial con ChromaDB).

ChromaDB guarda, por cada chunk:
    id        -> identificador único
    embedding -> el vector numérico
    document  -> el texto original del chunk
    metadata  -> diccionario (source, page, start_index...)

Usamos la integración `langchain_chroma.Chroma`, que implementa la interfaz
`VectorStore` de LangChain. Eso significa que:
- `add_documents()` calcula los embeddings automáticamente con el
  `embedding_function` que le pasamos.
- `similarity_search()` vectoriza la consulta y busca los vecinos más cercanos.
- Podríamos cambiar Chroma por FAISS, PGVector, Pinecone... sin tocar el
  agente, porque este solo depende de la interfaz `VectorStore`.

Para operaciones que no forman parte de esa interfaz común (listar por
metadata, borrar por filtro) usamos `.get()`, que es específico de Chroma.
"""

from collections import Counter
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from myfirstrag.config import settings


class VectorDatabase:
    def __init__(self, embeddings: Embeddings):
        self.store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            # Con persist_directory los datos sobreviven entre ejecuciones.
            persist_directory=str(settings.chroma_dir),
        )

    @staticmethod
    def _chunk_id(chunk: Document) -> str:
        """
        ID determinista por chunk (hash de fuente + página + posición).
        Si se vuelve a indexar el mismo archivo, los IDs coinciden y Chroma
        sobrescribe en lugar de duplicar.
        """
        m = chunk.metadata
        key = f"{m.get('source')}|{m.get('page')}|{m.get('start_index')}"
        return hashlib.sha256(key.encode()).hexdigest()

    def add_chunks(self, chunks: list[Document]) -> int:
        """Guarda los chunks (LangChain calcula los embeddings por debajo)."""
        if not chunks:
            return 0
        ids = [self._chunk_id(c) for c in chunks]
        self.store.add_documents(chunks, ids=ids)
        return len(chunks)

    def list_sources(self) -> dict[str, int]:
        """Devuelve {nombre_de_archivo: número_de_chunks}."""
        # include=["metadatas"] evita traer textos y vectores (más rápido).
        data = self.store.get(include=["metadatas"])
        return dict(Counter(m.get("source", "¿?") for m in data["metadatas"]))

    def delete_source(self, source: str) -> int:
        """Elimina todos los chunks de un archivo usando un filtro por metadata."""
        ids = self.store.get(where={"source": source}, include=[])["ids"]
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        return self.store.similarity_search(query, k=k or settings.top_k)
