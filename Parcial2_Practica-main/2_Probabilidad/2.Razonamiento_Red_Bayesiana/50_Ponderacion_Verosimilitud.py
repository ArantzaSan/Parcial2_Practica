import random

class Enigma:
    def __init__(self, rutas):
        """
        Inicializa el enigma con un diccionario de rutas.
        Cada clave es un punto y su valor es una lista de diccionarios:
        {'destino': punto_destino, 'probabilidad': probabilidad_transicion}.
        """
        self.rutas = rutas

    def estimar_probabilidad_destino(self, inicio, final, num_simulaciones=1000):
        """
        Estima la probabilidad de alcanzar el punto final desde el inicio
        utilizando muestreo con ponderación de verosimilitud.
        """
        llegadas_ponderadas = 0.0

        for _ in range(num_simulaciones):
            ubicacion_actual = inicio
            peso_trayectoria = 1.0

            while ubicacion_actual in self.rutas:
                opciones = self.rutas[ubicacion_actual]
                if not opciones:
                    break

                destino_elegido = random.choices(
                    opciones,
                    weights=[op['probabilidad'] for op in opciones]
                )[0]

                peso_trayectoria *= destino_elegido['probabilidad']
                ubicacion_actual = destino_elegido['destino']

                if ubicacion_actual == final:
                    llegadas_ponderadas += peso_trayectoria
                    break

        return llegadas_ponderadas / num_simulaciones


# Definición del enigma (análogo al laberinto)
mapa_rutas = {
    'A': [{'destino': 'B', 'probabilidad': 0.5}, {'destino': 'C', 'probabilidad': 0.5}],
    'B': [{'destino': 'D', 'probabilidad': 0.7}, {'destino': 'E', 'probabilidad': 0.3}],
    'C': [{'destino': 'E', 'probabilidad': 0.6}, {'destino': 'F', 'probabilidad': 0.4}],
    'D': [{'destino': 'G', 'probabilidad': 1.0}],
    'E': [{'destino': 'G', 'probabilidad': 0.8}, {'destino': 'H', 'probabilidad': 0.2}],
    'F': [{'destino': 'H', 'probabilidad': 1.0}],
    'G': [],
    'H': []
}

enigma = Enigma(mapa_rutas)

# Calcular la probabilidad de llegar de 'A' a 'G'
probabilidad_final = enigma.estimar_probabilidad_destino('A', 'G')
print(f"La probabilidad estimada de llegar de 'A' a 'G' es aproximadamente: {probabilidad_final:.4f}")
