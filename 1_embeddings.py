"""
Utilidad del script:

Este programa demuestra la base de un sistema RAG: convertir textos en embeddings,
es decir, vectores numéricos que representan su significado semántico.

Después compara esos vectores mediante similitud coseno para comprobar qué textos
son más parecidos entre sí, aunque no usen exactamente las mismas palabras.

También implementa una búsqueda semántica sencilla: dada una consulta, calcula su
embedding y ordena los textos según su cercanía semántica.

Este ejemplo sirve como primer paso antes de usar una vector store como Chroma,
FAISS o Pinecone, donde se almacenarían muchos documentos para recuperarlos
después en función de su relevancia.
"""


import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    Cuanto más cerca esté de 1, más parecidos son semánticamente.
    """
    a = np.array(vector_a)
    b = np.array(vector_b)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    load_dotenv()

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    text1 = "The cat sat on the mat"
    text2 = "A feline rested on the carpet"
    text3 = "Python is a programming language"

    texts = [text1, text2, text3]

    embeddings = embeddings_model.embed_documents(texts)

    embedding1 = embeddings[0]
    embedding2 = embeddings[1]
    embedding3 = embeddings[2]

    print("=== Información básica ===")
    print(f"Number of documents: {len(embeddings)}")
    print(f"Dimensions per embedding: {len(embeddings[0])}")

    print("\n=== Primeros 10 valores del embedding 1 ===")
    print(embedding1[:10])

    print("\n=== Similitudes entre textos ===")
    similarity_1_2 = cosine_similarity(embedding1, embedding2)
    similarity_1_3 = cosine_similarity(embedding1, embedding3)
    similarity_2_3 = cosine_similarity(embedding2, embedding3)

    print(f"text1 vs text2: {similarity_1_2:.4f}")
    print(f"text1 vs text3: {similarity_1_3:.4f}")
    print(f"text2 vs text3: {similarity_2_3:.4f}")

    """
    Quizás aquí podríamos tener un pequeño proyecto de buscadores
    """
    print("\n=== Búsqueda semántica simple ===")
    query = "A cat is sleeping on a rug"
    query_embedding = embeddings_model.embed_query(query)

    results = []

    for text, embedding in zip(texts, embeddings):
        score = cosine_similarity(query_embedding, embedding)
        results.append((text, score))

    results.sort(key=lambda item: item[1], reverse=True)

    print(f"Query: {query}\n")

    for text, score in results:
        print(f"{score:.4f} -> {text}")


if __name__ == "__main__":
    main()