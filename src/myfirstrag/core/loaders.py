"""
Paso 1 del pipeline — LOAD (carga).

En LangChain toda fuente de datos se normaliza a objetos `Document`:

    Document(page_content="texto...", metadata={"source": ..., "page": ...})

- `page_content`: el texto que se va a fragmentar, vectorizar y buscar.
- `metadata`: información extra (archivo de origen, página...). Se guarda en
  Chroma junto al vector y nos permite listar, filtrar y citar las fuentes.

LangChain tiene cientos de "document loaders" en `langchain_community`; aquí
escribimos uno mínimo con `pypdf` para ver que no hay magia detrás.
"""

from pathlib import Path

import pypdf
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_pdf(path: Path) -> list[Document]:
    """Un `Document` por página, guardando el número de página en metadata."""
    reader = pypdf.PdfReader(path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


def load_text(path: Path) -> list[Document]:
    """Archivos de texto plano / markdown: un único `Document`."""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8", errors="ignore"),
            metadata={"source": path.name, "page": 0},
        )
    ]


def load_document(path: Path) -> list[Document]:
    """Elige el loader adecuado según la extensión del archivo."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = load_pdf(path)
    elif ext in {".txt", ".md"}:
        docs = load_text(path)
    else:
        raise ValueError(f"Extensión no soportada: {ext}")
    # Descartamos páginas vacías (p. ej. PDFs escaneados sin texto).
    return [d for d in docs if d.page_content.strip()]
