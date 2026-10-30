"""
Interfaz gráfica con Streamlit.

Esta capa SOLO se encarga de mostrar cosas y recoger acciones del usuario.
Toda la lógica vive en `RAGService` (capa services), así que aquí no aparece
ningún import de LangChain ni de Chroma.

Ejecutar:
    uv run streamlit run src/myfirstrag/ui/app.py
    # o bien
    uv run myfirstrag

Recordatorio de cómo funciona Streamlit: el script se re-ejecuta COMPLETO de
arriba a abajo en cada interacción (clic, mensaje...). Por eso:
- `st.cache_resource` evita recrear objetos caros (cliente de Chroma, agente).
- `st.session_state` guarda datos que deben persistir entre re-ejecuciones
  (el historial del chat).
"""

import streamlit as st

from myfirstrag.config import settings
from myfirstrag.core.loaders import SUPPORTED_EXTENSIONS
from myfirstrag.services import RAGService

st.set_page_config(page_title="myFirstRAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Inicializando base vectorial y agente...")
def get_service() -> RAGService:
    return RAGService()


service = get_service()

# ---------------------------------------------------------------------- #
# Barra lateral: gestión de documentos
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("📂 Documentos")

    # --- Agregar documentos ---------------------------------------------
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader(
            "Agregar a la base de conocimiento",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        if st.form_submit_button("Indexar", type="primary") and files:
            for f in files:
                with st.spinner(f"Indexando {f.name}..."):
                    try:
                        n = service.ingest_upload(f.name, f.getvalue())
                        st.success(f"{f.name}: {n} chunks")
                    except Exception as e:
                        st.error(f"{f.name}: {e}")

    # --- Archivos en docs/ aún no indexados -----------------------------
    indexed = service.list_documents()
    pending = sorted(
        p
        for p in settings.docs_dir.glob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.name not in indexed
    )
    if pending:
        st.caption(f"{len(pending)} archivo(s) en `docs/` sin indexar")
        if st.button("Indexar archivos de docs/"):
            for p in pending:
                with st.spinner(f"Indexando {p.name}..."):
                    service.ingest_file(p)
            st.rerun()

    # --- Listado de la base vectorial -----------------------------------
    st.subheader("En la base vectorial")
    if not indexed:
        st.info("La colección está vacía.")
    for source, n_chunks in sorted(indexed.items()):
        col_name, col_btn = st.columns([5, 1])
        col_name.markdown(f"**{source}**  \n{n_chunks} chunks")
        if col_btn.button("🗑️", key=f"del-{source}", help="Eliminar"):
            service.delete_document(source)
            st.rerun()

    st.divider()
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------- #
# Área principal: chat con el agente RAG
# ---------------------------------------------------------------------- #
st.title("📚 myFirstRAG")
st.caption(
    f"Búsqueda semántica con LangChain + ChromaDB · modelo `{settings.chat_model}`"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Fuentes consultadas ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}** — pág. {s['page'] + 1}")
            st.text(s["content"][:500])


# Repinta el historial (Streamlit re-ejecuta todo en cada interacción).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

if question := st.chat_input("Pregunta algo sobre tus documentos..."):
    with st.chat_message("user"):
        st.markdown(question)

    # Historial en el formato que espera el servicio (sin las fuentes).
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos..."):
            try:
                answer = service.ask(question, history)
                text = answer.text
                sources = [
                    {
                        "source": d.metadata.get("source", "¿?"),
                        "page": d.metadata.get("page", 0),
                        "content": d.page_content,
                    }
                    for d in answer.sources
                ]
            except Exception as e:
                text, sources = f"⚠️ Error: {e}", []
        st.markdown(text)
        render_sources(sources)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text, "sources": sources},
    ]
