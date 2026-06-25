"""
Ejemplo básico de Vector Store con LangChain y Chroma.

Este script demuestra cómo:
1. Crear documentos con contenido y metadatos.
2. Convertir esos documentos en embeddings usando OpenAIEmbeddings.
3. Guardarlos en una vector store local con Chroma.
4. Buscar documentos similares a una consulta.
5. Buscar documentos similares con puntuación.
6. Usar MMR para recuperar documentos relevantes pero menos redundantes.

Este es uno de los pasos fundamentales de un sistema RAG:
documentos -> embeddings -> vector store -> recuperación semántica.
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def print_results(title: str, results: list[Document]) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    for index, doc in enumerate(results, start=1):
        print(f"\nResultado {index}")
        print(f"Contenido: {doc.page_content}")
        print(f"Metadatos: {doc.metadata}")


def main() -> None:
    load_dotenv()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    documents = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "docs_langchain", "page": 1, "topic": "langchain"},
        ),
        Document(
            page_content="Chroma is a lightweight vector database useful for local RAG prototypes.",
            metadata={"source": "docs_chroma", "page": 2, "topic": "vector_store"},
        ),
        Document(
            page_content="Retrieval-Augmented Generation combines document retrieval with language model generation.",
            metadata={"source": "rag_notes", "page": 3, "topic": "rag"},
        ),
        Document(
            page_content="Python is a programming language commonly used for data science and AI applications.",
            metadata={"source": "python_notes", "page": 4, "topic": "python"},
        ),
        Document(
            page_content="Vector stores allow semantic search by comparing embeddings in high-dimensional space.",
            metadata={"source": "vector_notes", "page": 5, "topic": "vector_store"},
        ),
    ]

    vector_store = Chroma(
        collection_name="langchain_rag_demo",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    ids = vector_store.add_documents(documents)

    print("\nDocumentos añadidos a Chroma:")
    for doc_id in ids:
        print(f"- {doc_id}")

    query = "How can I store embeddings for semantic search?"

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    print_results(
        title=f"Búsqueda por similitud para: {query}",
        results=results,
    )

    results_with_scores = vector_store.similarity_search_with_score(
        query=query,
        k=3,
    )

    print(f"\n{'=' * 80}")
    print("Búsqueda por similitud con puntuación")
    print("=" * 80)

    for index, (doc, score) in enumerate(results_with_scores, start=1):
        print(f"\nResultado {index}")
        print(f"Score: {score}")
        print(f"Contenido: {doc.page_content}")
        print(f"Metadatos: {doc.metadata}")

    mmr_results = vector_store.max_marginal_relevance_search(
        query=query,
        k=3,
        fetch_k=5,
        lambda_mult=0.5,
    )

    print_results(
        title="Búsqueda MMR: relevante pero reduciendo redundancia",
        results=mmr_results,
    )

    # Ejemplo de borrado de documentos.
    # Descomenta si quieres eliminar los documentos insertados.
    #
    # vector_store.delete(ids=ids)
    # print("Documentos eliminados.")


if __name__ == "__main__":
    main()