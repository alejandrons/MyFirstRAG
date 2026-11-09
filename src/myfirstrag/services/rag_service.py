"""
Servicio RAG: el "pegamento" entre la capa LangChain (core), la base de datos
vectorial (database) y cualquier interfaz (Streamlit, CLI, API REST...).

La UI solo llama a estos métodos; no importa nada de LangChain ni de Chroma.
Así, si mañana cambiamos Streamlit por FastAPI, este archivo no cambia.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from myfirstrag.config import settings
from myfirstrag.core.agent import build_rag_agent, ping_chat_model
from myfirstrag.core.embeddings import get_embeddings
from myfirstrag.core.loaders import SUPPORTED_EXTENSIONS, load_document
from myfirstrag.core.splitter import split_documents
from myfirstrag.database import VectorDatabase


@dataclass
class Answer:
    text: str
    sources: list[Document] = field(default_factory=list)


@dataclass
class ConnectionCheck:
    ok: bool
    model: str
    message: str
    latency_s: float = 0.0


class RAGService:
    def __init__(self):
        self.db = VectorDatabase(get_embeddings())
        # Un agente por modelo de chat, creado bajo demanda y reutilizado.
        # La base vectorial (y los embeddings) es la misma para todos: cambiar
        # de LLM no obliga a reindexar nada.
        self._agents: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Modelos de chat
    # ------------------------------------------------------------------ #
    def get_agent(self, model_name: str | None = None):
        model_name = model_name or settings.chat_model
        if model_name not in self._agents:
            # El agente recibe el VectorStore de LangChain (no nuestro wrapper):
            # solo necesita la interfaz estándar `similarity_search`.
            self._agents[model_name] = build_rag_agent(self.db.store, model_name)
        return self._agents[model_name]

    def test_connection(self, model_name: str) -> ConnectionCheck:
        """Hace una llamada mínima al modelo para verificar que responde."""
        start = time.perf_counter()
        try:
            reply = ping_chat_model(model_name)
        except Exception as e:
            return ConnectionCheck(ok=False, model=model_name, message=str(e))
        return ConnectionCheck(
            ok=True,
            model=model_name,
            message=reply.strip(),
            latency_s=time.perf_counter() - start,
        )

    # ------------------------------------------------------------------ #
    # Indexación: Load -> Split -> Embed + Store
    # ------------------------------------------------------------------ #
    def ingest_file(self, path: Path) -> int:
        """Indexa un archivo del disco. Devuelve el número de chunks creados."""
        docs = load_document(path)          # 1. Load
        chunks = split_documents(docs)      # 2. Split
        # Si el archivo ya estaba indexado, borramos su versión anterior para
        # no dejar chunks huérfanos si el contenido cambió.
        self.db.delete_source(path.name)
        return self.db.add_chunks(chunks)   # 3. Embed + 4. Store

    def ingest_upload(self, filename: str, content: bytes) -> int:
        """Guarda un archivo subido desde la UI en `docs/` y lo indexa."""
        # Path(...).name descarta rutas como "../../etc/passwd".
        safe_name = Path(filename).name
        if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Formato no soportado: {safe_name}")
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        path = settings.docs_dir / safe_name
        path.write_bytes(content)
        return self.ingest_file(path)

    # ------------------------------------------------------------------ #
    # Gestión de la colección
    # ------------------------------------------------------------------ #
    def list_documents(self) -> dict[str, int]:
        return self.db.list_sources()

    def delete_document(self, source: str) -> int:
        return self.db.delete_source(source)

    # ------------------------------------------------------------------ #
    # Consulta: Retrieve + Generate
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        model_name: str | None = None,
    ) -> Answer:
        """
        Envía la pregunta (más el historial del chat) al agente del modelo
        `model_name` (por defecto, `settings.chat_model`).

        `history` es una lista de {"role": "user"|"assistant", "content": str};
        LangChain acepta mensajes en ese formato de diccionario directamente.
        """
        messages = [*(history or []), {"role": "user", "content": question}]
        result = self.get_agent(model_name).invoke({"messages": messages})

        # result["messages"] contiene TODA la conversación, incluidos los
        # mensajes intermedios del bucle agente:
        #   HumanMessage -> AIMessage(tool_calls) -> ToolMessage -> AIMessage
        new_messages = result["messages"][len(messages):]
        sources = [
            doc
            for m in new_messages
            if isinstance(m, ToolMessage) and m.artifact
            for doc in m.artifact
        ]
        return Answer(text=result["messages"][-1].text, sources=sources)
