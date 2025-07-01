from typing import List

# Importamos las bibliotecas necesarias

# Definimos una clase para representar las proposiciones modales
class ModalProposition:
    def __init__(self, proposition: str):
        self.proposition = proposition

    def __str__(self):
        return self.proposition

# Definimos una clase para representar las operaciones modales
class ModalLogic:
    @staticmethod
    def necessary(proposition: ModalProposition) -> str:
        # Operador de necesidad (□)
        return f"□({proposition})"

    @staticmethod
    def possible(proposition: ModalProposition) -> str:
        # Operador de posibilidad (◇)
        return f"◇({proposition})"

# Clase para analizar progresiones musicales usando lógica modal
class MusicalAnalysis:
    def __init__(self, chords: List[str]):
        self.chords = chords

    def analyze_progression(self):
        # Analizamos la progresión de acordes usando lógica modal
        analysis = []
        for chord in self.chords:
            proposition = ModalProposition(f"El acorde {chord} es tonalmente estable")
            necessary = ModalLogic.necessary(proposition)
            possible = ModalLogic.possible(proposition)
            analysis.append((necessary, possible))
        return analysis

# Ejemplo de uso del programa
if __name__ == "__main__":
    # Definimos una progresión de acordes
    chord_progression = ["C", "G", "Am", "F"]

    # Creamos una instancia de análisis musical
    analyzer = MusicalAnalysis(chord_progression)

    # Analizamos la progresión
    results = analyzer.analyze_progression()

    # Mostramos los resultados
    for necessary, possible in results:
        print(f"Necesario: {necessary}")
        print(f"Posible: {possible}")
        print()
