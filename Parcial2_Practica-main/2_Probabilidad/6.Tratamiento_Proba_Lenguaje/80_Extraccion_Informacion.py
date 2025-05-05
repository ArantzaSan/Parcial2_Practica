import spacy

def extraer_entidades(texto):
    """Extrae entidades nombradas de un texto usando spaCy."""
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(texto)
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    return entidades

def extraer_relaciones_tripletas(texto):
    """Intenta extraer tripletas sujeto-verbo-objeto simples."""
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(texto)
    tripletas = []
    for token in doc:
        if token.dep_ == "ROOT":  # Buscar el verbo principal
            sujeto = [t.text for t in token.lefts if t.dep_ in ["nsubj", "nsubjpass"]]
            objeto = [t.text for t in token.rights if t.dep_ in ["dobj", "pobj"]]
            if sujeto and objeto:
                tripletas.append((sujeto[0], token.text, objeto[0]))
    return tripletas

# Ejemplo de texto
texto_ejemplo = "Apple fue fundada por Steve Jobs en California en 1976. Ahora tiene su sede central en Cupertino."

# Extracción de entidades
entidades_encontradas = extraer_entidades(texto_ejemplo)
print("Entidades encontradas:", entidades_encontradas)

# Extracción de relaciones (simple)
tripletas_encontradas = extraer_relaciones_tripletas(texto_ejemplo)
print("Tripletas encontradas (simple):", tripletas_encontradas)
