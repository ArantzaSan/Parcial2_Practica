from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recuperar_datos(corpus, consulta, top_n=3):
    """
    Recupera los documentos más relevantes del corpus para una consulta dada.

    Args:
        corpus (list): Lista de cadenas de texto (documentos).
        consulta (str): Cadena de texto de la consulta.
        top_n (int): Número de documentos más relevantes a devolver.

    Returns:
        list: Lista de los top_n documentos más relevantes, ordenados por similitud descendente.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [consulta])
    consulta_vector = tfidf_matrix[-1]
    corpus_vectors = tfidf_matrix[:-1]

    similarity_scores = cosine_similarity(consulta_vector, corpus_vectors)[0]
    ranked_documents = sorted(zip(corpus, similarity_scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked_documents[:top_n]]

# Ejemplo de uso
corpus_ejemplo = [
    "El perro ladra fuerte por la noche.",
    "Un gato duerme plácidamente al sol.",
    "Los pájaros cantan melodías al amanecer.",
    "La noche es oscura y llena de terrores.",
    "El sol brilla intensamente en el cielo azul."
]

consulta_ejemplo = "animales que hacen ruido por la noche"

resultados = recuperar_datos(corpus_ejemplo, consulta_ejemplo)

print(f"Consulta: '{consulta_ejemplo}'")
print(f"Top {len(resultados)} resultados:")
for i, resultado in enumerate(resultados):
    print(f"{i+1}. {resultado}")
