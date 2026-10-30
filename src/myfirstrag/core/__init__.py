"""
Capa `core`: todo el uso de LangChain para el pipeline RAG.

Cada módulo corresponde a un paso del pipeline:

    loaders.py     -> 1. Load   : archivo -> list[Document]
    splitter.py    -> 2. Split  : Document grande -> chunks pequeños
    embeddings.py  -> 3. Embed  : texto -> vector numérico
    (database/)    -> 4. Store  : vectores -> ChromaDB
    agent.py       -> 5. Retrieve + Generate : agente con herramienta de búsqueda

Esta capa NO sabe nada de Streamlit ni de cómo se guardan los archivos:
solo transforma datos usando las abstracciones de LangChain.
"""
