import random

# Catálogo musical con atributos clave
catalogo_musical = [
    {"titulo": "Melodía Serena", "animo": "tranquilo", "estilo": "instrumental", "fama": 0.8},
    {"titulo": "Ritmo Vibrante", "animo": "animado", "estilo": "electronica", "fama": 0.9},
    {"titulo": "Tonada Alegre", "animo": "contento", "estilo": "indie", "fama": 0.7},
    {"titulo": "Canto Nostálgico", "animo": "melancolico", "estilo": "folk", "fama": 0.6},
    {"titulo": "Sonido Ambiental", "animo": "tranquilo", "estilo": "ambiental", "fama": 0.5},
]

def calcular_preferencia(cancion, gustos_usuario, situacion_actual):
    # Función de preferencia basada en los gustos del usuario y el contexto
    puntuacion = 0
    if cancion["animo"] == gustos_usuario["animo"]:
        puntuacion += 3
    if cancion["estilo"] == gustos_usuario["estilo"]:
        puntuacion += 3
    if situacion_actual["momento_dia"] == "noche" and cancion["animo"] == "tranquilo":
        puntuacion += 2
    if situacion_actual["momento_dia"] == "dia" and cancion["animo"] == "animado":
        puntuacion += 2
    puntuacion += cancion["fama"]
    return puntuacion

def sistema_recomendacion_musical(gustos_usuario, situacion_actual):
    # Evaluar cada canción del catálogo
    canciones_valoradas = [(cancion, calcular_preferencia(cancion, gustos_usuario, situacion_actual)) for cancion in catalogo_musical]

    # Seleccionar la canción con la mayor valoración
    mejor_opcion = max(canciones_valoradas, key=lambda item: item[1])
    return mejor_opcion[0]

# Ajustes del usuario
preferencias_usuario = {
    "animo": "tranquilo",
    "estilo": "instrumental"
}

# Escenario actual
contexto_actual = {
    "momento_dia": "noche"
}

# Obtener la recomendación musical
cancion_sugerida = sistema_recomendacion_musical(preferencias_usuario, contexto_actual)
print(f"Sugerencia musical: {cancion_sugerida['titulo']} (Estilo: {cancion_sugerida['estilo']}, Ánimo: {cancion_sugerida['animo']})")
