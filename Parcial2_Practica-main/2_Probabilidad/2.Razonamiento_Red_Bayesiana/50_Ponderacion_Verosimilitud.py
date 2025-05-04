import random

class Laberinto:
    def __init__(self, caminos):
        """
        Inicializa el laberinto con un diccionario de caminos.
        Cada clave es un nodo y su valor es una lista de tuplas (nodo_destino, probabilidad).
        """
        self.caminos = caminos

    def ponderacion_verosimilitud(self, inicio, destino, num_simulaciones=1000):
        """
        Calcula la probabilidad de llegar al destino desde el inicio usando ponderación de verosimilitud.
        """
        exitos = 0

        for _ in range(num_simulaciones):
            nodo_actual = inicio
            peso_acumulado = 1.0

            while nodo_actual in self.caminos:
                siguiente_nodo, probabilidad = random.choices(
                    self.caminos[nodo_actual],
                    weights=[p for _, p in self.caminos[nodo_actual]]
                )[0]

                peso_acumulado *= probabilidad
                nodo_actual = siguiente_nodo

                if nodo_actual == destino:
                    exitos += peso_acumulado
                    break

        return exitos / num_simulaciones


# Definición del laberinto
caminos = {
    'A': [('B', 0.5), ('C', 0.5)],
    'B': [('D', 0.7), ('E', 0.3)],
    'C': [('E', 0.6), ('F', 0.4)],
    'D': [('G', 1.0)],
    'E': [('G', 0.8), ('H', 0.2)],
    'F': [('H', 1.0)],
    'G': [],
    'H': []
}

laberinto = Laberinto(caminos)

# Calcular la probabilidad de llegar de 'A' a 'G'
probabilidad = laberinto.ponderacion_verosimilitud('A', 'G')
print(f"La probabilidad de llegar de 'A' a 'G' es aproximadamente: {probabilidad:.4f}")