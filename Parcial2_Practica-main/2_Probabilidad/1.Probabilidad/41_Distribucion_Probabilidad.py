import numpy as np
import matplotlib.pyplot as plt

class DistribucionArticulos:
    def __init__(self, probabilidades_articulo):
        self.probabilidades_articulo = probabilidades_articulo  # Probabilidades de elección de cada bien
        self.articulos = list(probabilidades_articulo.keys())

    def simular_elecciones(self, num_jugadores):
        # Simular las elecciones de los jugadores
        elecciones = np.random.choice(self.articulos, size=num_jugadores, p=list(self.probabilidades_articulo.values()))
        return elecciones

    def graficar_distribucion(self, elecciones):
        # Graficar la distribución de elecciones
        unicos, conteos = np.unique(elecciones, return_counts=True)
        plt.bar(unicos, conteos, color='lightcoral')
        plt.xlabel('Artículos')
        plt.ylabel('Número de Jugadores')
        plt.title('Distribución de Elección de Artículos')
        plt.show()

# Probabilidades de elección de cada bien
probabilidades_articulo = {
    'Espada': 0.5,
    'Arco': 0.3,
    'Escudo': 0.2
}

# Crear el modelo de distribución de bienes
distribucion_articulos = DistribucionArticulos(probabilidades_articulo)

# Simular las elecciones de 1000 jugadores
numero_jugadores = 1000
elecciones_simuladas = distribucion_articulos.simular_elecciones(numero_jugadores)

# Graficar la distribución de elecciones
distribucion_articulos.graficar_distribucion(elecciones_simuladas)

# Mostrar la distribución de elecciones
unicos, conteos = np.unique(elecciones_simuladas, return_counts=True)
distribucion_probabilidad = dict(zip(unicos, conteos / numero_jugadores))
print("Distribución de probabilidad de elección de artículos:")
for articulo, probabilidad in distribucion_probabilidad.items():
    print(f"{articulo}: {probabilidad:.2f}")
