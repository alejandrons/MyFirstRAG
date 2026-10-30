"""
Capa `database`: acceso a la base de datos vectorial (ChromaDB).

Encapsula todas las operaciones sobre Chroma (agregar, listar, borrar,
buscar) para que el resto del proyecto no dependa de sus detalles.
"""

from myfirstrag.database.vector_store import VectorDatabase

__all__ = ["VectorDatabase"]
