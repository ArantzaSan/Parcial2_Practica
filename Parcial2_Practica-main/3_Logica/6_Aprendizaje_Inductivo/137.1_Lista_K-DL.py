# 137.1_Lista_K-DL.py
# Implementación de una lista K-DT (K-Decision List) orientada a música
# Una K-Decision List es una secuencia de reglas condicionales para clasificar ejemplos

# Definimos una clase para representar una regla de decisión
class ReglaDecision:
    def __init__(self, condicion, resultado):
        """
        condicion: función que recibe una canción y retorna True/False
        resultado: valor de clasificación si la condición es verdadera
        """
        self.condicion = condicion
        self.resultado = resultado

# Clase para la lista K-DT
class ListaKDT:
    def __init__(self):
        self.reglas = []

    def agregar_regla(self, condicion, resultado):
        """
        Agrega una nueva regla a la lista
        """
        self.reglas.append(ReglaDecision(condicion, resultado))

    def clasificar(self, cancion):
        """
        Clasifica una canción según las reglas de la lista K-DT
        """
        for regla in self.reglas:
            if regla.condicion(cancion):
                return regla.resultado
        return None  # Si ninguna regla aplica

# Ejemplo de uso orientado a música

# Definimos algunas canciones como diccionarios
canciones = [
    {"titulo": "Song A", "genero": "rock", "bpm": 120, "instrumental": False},
    {"titulo": "Song B", "genero": "clasica", "bpm": 70, "instrumental": True},
    {"titulo": "Song C", "genero": "pop", "bpm": 100, "instrumental": False},
    {"titulo": "Song D", "genero": "jazz", "bpm": 90, "instrumental": True},
]

# Creamos una lista K-DT para clasificar canciones como "Energética" o "Relajante"
lista_kdt = ListaKDT()

# Agregamos reglas a la lista
# Regla 1: Si la canción es rock y tiene más de 110 bpm, es "Energética"
lista_kdt.agregar_regla(
    lambda c: c["genero"] == "rock" and c["bpm"] > 110,
    "Energética"
)

# Regla 2: Si la canción es instrumental y tiene menos de 80 bpm, es "Relajante"
lista_kdt.agregar_regla(
    lambda c: c["instrumental"] and c["bpm"] < 80,
    "Relajante"
)

# Regla 3: Si la canción es pop, es "Energética"
lista_kdt.agregar_regla(
    lambda c: c["genero"] == "pop",
    "Energética"
)

# Regla 4: Si la canción es jazz, es "Relajante"
lista_kdt.agregar_regla(
    lambda c: c["genero"] == "jazz",
    "Relajante"
)

# Clasificamos las canciones y mostramos los resultados
for cancion in canciones:
    clasificacion = lista_kdt.clasificar(cancion)
    print(f'{cancion["titulo"]}: {clasificacion}')

# Fin del programa
