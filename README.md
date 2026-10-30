# 📚 myFirstRAG

Proyecto educativo para aprender a construir un sistema **RAG (Retrieval Augmented Generation)** con **LangChain**, una base de datos vectorial **ChromaDB** y una interfaz gráfica en **Streamlit**.

Desde la UI puedes:

- **Agregar documentos** (PDF, TXT, MD) a la base de conocimiento.
- **Listar y eliminar** los documentos que hay en la base vectorial.
- **Hacer preguntas** a un agente que busca en los documentos y responde citando las fuentes.

---

## 🚀 Puesta en marcha

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.14.

```bash
uv sync
```

```bash
cp .env.example .env
```

Edita `.env` y añade tu `OPENAI_API_KEY`. Después lanza la UI:

```bash
uv run myfirstrag
```

(equivale a `uv run streamlit run src/myfirstrag/ui/app.py`). Se abrirá en http://localhost:8501.

> Los archivos que ya estén en `docs/` pero no en la base vectorial aparecen en la barra lateral con un botón **"Indexar archivos de docs/"**.

---

## 🧠 ¿Qué es un RAG?

Un LLM solo sabe lo que vio durante su entrenamiento. Un RAG le da acceso a **tus** documentos en dos fases:

```
 FASE 1 · INDEXACIÓN (una vez por documento)
 ┌────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────┐
 │  Load  │──►│  Split  │──►│  Embed   │──►│    Store    │
 │ PDF →  │   │ páginas │   │ texto →  │   │  ChromaDB   │
 │Document│   │→ chunks │   │ vectores │   │  (en disco) │
 └────────┘   └─────────┘   └──────────┘   └─────────────┘

 FASE 2 · CONSULTA (cada pregunta)
 pregunta ──► Agente (LLM) ──► tool search_documents ──► ChromaDB
                  ▲                                         │
                  └──────── chunks más similares ◄──────────┘
                  │
                  └──► respuesta con citas
```

1. **Load**: se lee el archivo y se convierte en objetos `Document` (texto + metadata).
2. **Split**: cada página se parte en *chunks* de ~1000 caracteres; trozos pequeños hacen la búsqueda más precisa.
3. **Embed**: un modelo de *embeddings* convierte cada chunk en un vector. Textos con significado parecido → vectores cercanos.
4. **Store**: los vectores se guardan en ChromaDB junto con el texto y la metadata.
5. **Retrieve + Generate**: la pregunta se vectoriza, se buscan los chunks más cercanos (búsqueda semántica) y el LLM redacta la respuesta usando solo ese contexto.

---

## 🏗️ Arquitectura del proyecto

```
myFirstRAG/
├── docs/                         # Archivos fuente (los subidos desde la UI se guardan aquí)
├── notebooks/
│   └── 01_rag_pipeline.ipynb     # Exploración paso a paso del pipeline
├── chroma_langchain_db/          # Base vectorial persistida (ignorada por git)
└── src/myfirstrag/
    ├── config.py                 # Configuración central (.env, rutas, modelos)
    ├── core/                     # ⭐ Capa LangChain: el pipeline RAG
    │   ├── loaders.py            #    1. Load   → Document
    │   ├── splitter.py           #    2. Split  → RecursiveCharacterTextSplitter
    │   ├── embeddings.py         #    3. Embed  → OpenAIEmbeddings
    │   └── agent.py              #    5. Retrieve + Generate → @tool + create_agent
    ├── database/                 # Capa de datos
    │   └── vector_store.py       #    4. Store  → ChromaDB (añadir, listar, borrar, buscar)
    ├── services/                 # Capa de orquestación
    │   └── rag_service.py        #    Une core + database en una API sencilla
    └── ui/                       # Capa de presentación
        └── app.py                #    Streamlit (no importa LangChain)
```

### Dependencias entre capas

```
   ui  ──►  services  ──►  core      (LangChain: loaders, splitter, embeddings, agente)
                      └──►  database (LangChain + Chroma: almacenamiento vectorial)
```

Las flechas van en **un solo sentido**. Esto permite:

| Si quieres...                           | Solo cambias...                         |
| --------------------------------------- | --------------------------------------- |
| Usar FastAPI o una CLI en vez de Streamlit | `ui/`                                |
| Cambiar Chroma por FAISS / PGVector     | `database/vector_store.py`              |
| Usar embeddings de Google u Ollama      | `core/embeddings.py`                    |
| Usar Claude, Gemini u otro LLM          | `CHAT_MODEL` en `.env`                  |
| Soportar Word o HTML                    | `core/loaders.py`                       |

---

## 🦜 Conceptos de LangChain usados

| Concepto | Dónde | Para qué sirve |
| --- | --- | --- |
| `Document` | `core/loaders.py` | Unidad de datos estándar: `page_content` + `metadata`. Todo el ecosistema la entiende. |
| `RecursiveCharacterTextSplitter` | `core/splitter.py` | Corta texto respetando párrafos/líneas; `chunk_overlap` evita perder contexto entre chunks. |
| `Embeddings` (interfaz) / `OpenAIEmbeddings` | `core/embeddings.py` | Texto → vector. La interfaz común permite cambiar de proveedor sin tocar el resto. |
| `VectorStore` (interfaz) / `Chroma` | `database/vector_store.py` | Guarda vectores y hace `similarity_search`. `add_documents` calcula los embeddings automáticamente. |
| `init_chat_model` | `core/agent.py` | Crea un chat model desde un string `"proveedor:modelo"` (p. ej. `anthropic:claude-sonnet-5-5`). |
| `@tool` | `core/agent.py` | Convierte una función Python en una herramienta que el LLM puede invocar. El docstring es la "descripción" que lee el LLM. |
| `response_format="content_and_artifact"` | `core/agent.py` | La tool devuelve texto para el LLM **y** los `Document` originales para la UI (fuentes). |
| `create_agent` | `core/agent.py` | Construye el bucle agente (sobre LangGraph): LLM ↔ tools hasta tener la respuesta. |
| Mensajes (`HumanMessage`, `AIMessage`, `ToolMessage`) | `services/rag_service.py` | El estado del agente es una lista de mensajes; de los `ToolMessage` extraemos las fuentes. |

### ¿Por qué un agente y no una cadena fija?

Una cadena "buscar → responder" **siempre** busca, aunque digas "hola". El agente decide **cuándo** y **cuántas veces** buscar, y puede reformular la consulta si el primer resultado no basta. Es el paso previo a arquitecturas con subagentes (por ejemplo con `deepagents`, ya incluido en las dependencias).

---

## ⚙️ Configuración

Todas las opciones están en `src/myfirstrag/config.py` y se pueden sobrescribir desde `.env`:

| Variable | Por defecto | Efecto |
| --- | --- | --- |
| `CHAT_MODEL` | `openai:gpt-5-mini` | LLM del agente (`proveedor:modelo`) |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Modelo de embeddings de OpenAI |
| `CHROMA_COLLECTION` | `myfirstrag` | Nombre de la colección en Chroma |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Tamaño y solapamiento de los chunks |
| `TOP_K` | `4` | Chunks devueltos por cada búsqueda |

> ⚠️ Si cambias `EMBEDDING_MODEL`, usa otra colección o vuelve a indexar: los vectores de modelos distintos no son comparables.

---

## 🧪 Ideas para seguir aprendiendo

- Ajusta `CHUNK_SIZE` y `TOP_K` y observa cómo cambian las fuentes recuperadas.
- Usa `max_marginal_relevance_search` en lugar de `similarity_search` para obtener resultados más diversos.
- Filtra por documento pasando `filter={"source": "..."}` a la búsqueda.
- Añade memoria persistente al agente con un `checkpointer` de LangGraph.
- Implementa subagentes de análisis con `deepagents` (pasos 3 y 4 del notebook).
