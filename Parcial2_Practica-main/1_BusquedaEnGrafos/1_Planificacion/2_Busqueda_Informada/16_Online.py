import random

# Base de datos de canciones con características
songs_database = [
    {"title": "Canción Relajante", "mood": "relajante", "genre": "clásica", "popularity": 8},
    {"title": "Canción Energética", "mood": "energético", "genre": "rock", "popularity": 9},
    {"title": "Canción Feliz", "mood": "feliz", "genre": "pop", "popularity": 7},
    {"title": "Canción Triste", "mood": "triste", "genre": "balada", "popularity": 6},
    {"title": "Canción Nocturna", "mood": "relajante", "genre": "jazz", "popularity": 5},
]

def heuristic(song, user_preferences, context):
    # Heurística basada en las preferencias del usuario y el contexto actual
    score = 0
    if song["mood"] == user_preferences["mood"]:
        score += 2
    if song["genre"] == user_preferences["genre"]:
        score += 2
    if context["time_of_day"] == "noche" and song["mood"] == "relajante":
        score += 1
    if context["time_of_day"] == "día" and song["mood"] == "energético":
        score += 1
    score += song["popularity"] / 10
    return score

def online_music_recommendation(user_preferences, context):
    # Evaluar todas las canciones en la base de datos
    scored_songs = [(song, heuristic(song, user_preferences, context)) for song in songs_database]

    # Seleccionar la canción con la puntuación más alta
    best_song = max(scored_songs, key=lambda x: x[1])
    return best_song[0]

# Preferencias del usuario
user_preferences = {
    "mood": "relajante",
    "genre": "clásica"
}

# Contexto actual (puede cambiar en tiempo real)
context = {
    "time_of_day": "noche"
}

# Obtener recomendación de música
recommended_song = online_music_recommendation(user_preferences, context)
print(f"Canción recomendada: {recommended_song['title']} (Género: {recommended_song['genre']}, Estado de ánimo: {recommended_song['mood']})")
