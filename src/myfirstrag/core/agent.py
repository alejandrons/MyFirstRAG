"""
Paso 5 del pipeline — RETRIEVE + GENERATE (agente RAG).

En lugar de un pipeline fijo "buscar -> responder", usamos un AGENTE:
un LLM que puede decidir llamar herramientas (tools) en un bucle:

    pregunta ──► LLM ──¿necesito datos?──► tool search_documents ──┐
                  ▲                                                │
                  └──────────── resultados (chunks) ◄──────────────┘
                  │
                  └──► respuesta final (cuando ya tiene suficiente contexto)

Ventajas frente a un RAG "de un paso":
- Si la pregunta es un saludo, el agente no busca nada.
- Puede reformular la consulta y buscar varias veces.
- Es la base para sistemas más complejos (subagentes, `deepagents`...).

Piezas de LangChain usadas aquí:
- `init_chat_model`: crea un chat model a partir de "proveedor:modelo". Así
  cambiar de LLM (OpenAI, Anthropic, Google...) es solo cambiar un string.
- `@tool`: convierte una función Python en una herramienta que el LLM
  puede invocar. El docstring y los type hints forman el esquema que ve el LLM.
- `create_agent`: construye el bucle agente (sobre LangGraph) LLM <-> tools.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from myfirstrag.config import settings

SYSTEM_PROMPT = """Eres un asistente que responde preguntas sobre los documentos \
de una base de conocimiento.

- Usa SIEMPRE la herramienta `search_documents` antes de responder preguntas \
sobre el contenido de los documentos. Puedes llamarla varias veces con \
consultas distintas si la primera no es suficiente.
- Responde únicamente con información encontrada en los documentos. Si no \
encuentras la respuesta, dilo claramente; no inventes.
- Cita la fuente al final con el formato (archivo, pág. N).
- Responde en el mismo idioma de la pregunta."""


def format_docs(docs: list[Document]) -> str:
    """Serializa los chunks recuperados como texto legible para el LLM."""
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source')}, pág. {d.metadata.get('page', 0) + 1}]\n"
        f"{d.page_content}"
        for d in docs
    )


def build_retrieval_tool(vector_store: VectorStore):
    """
    Crea la herramienta de búsqueda ligada a un vector store concreto.

    Usamos una "closure" (función dentro de función) para que la tool tenga
    acceso al `vector_store` sin que el LLM tenga que pasarlo como argumento.
    """

    # response_format="content_and_artifact" permite devolver dos cosas:
    #   - content : el texto que verá el LLM.
    #   - artifact: objetos Python (los Document originales) que NO van al LLM
    #     pero quedan en el ToolMessage; la UI los usa para mostrar las fuentes.
    @tool(response_format="content_and_artifact")
    def search_documents(query: str) -> tuple[str, list[Document]]:
        """Busca en la base de conocimiento los fragmentos más relevantes para
        responder a la consulta. Usa palabras clave concretas de la pregunta."""
        # similarity_search: vectoriza `query` con el mismo modelo de embeddings
        # y devuelve los k chunks cuyos vectores están más cerca.
        docs = vector_store.similarity_search(query, k=settings.top_k)
        return format_docs(docs), docs

    return search_documents


def build_chat_model(model_name: str | None = None) -> BaseChatModel:
    """Crea el chat model a partir de "proveedor:modelo" (p. ej. "openai:gpt-5-mini")."""
    return init_chat_model(model_name or settings.chat_model)


def ping_chat_model(model_name: str) -> str:
    """
    Envía un mensaje mínimo al modelo para comprobar que la conexión funciona
    (paquete del proveedor instalado, API key válida, nombre de modelo correcto).
    Devuelve el texto de la respuesta; si algo falla, lanza la excepción.
    """
    model = build_chat_model(model_name)
    reply = model.invoke("Responde únicamente con la palabra: OK")
    return reply.text


def build_rag_agent(vector_store: VectorStore, model_name: str | None = None):
    """Ensambla LLM + herramienta + prompt de sistema en un agente ejecutable."""
    model = build_chat_model(model_name)
    return create_agent(
        model=model,
        tools=[build_retrieval_tool(vector_store)],
        system_prompt=SYSTEM_PROMPT,
    )
