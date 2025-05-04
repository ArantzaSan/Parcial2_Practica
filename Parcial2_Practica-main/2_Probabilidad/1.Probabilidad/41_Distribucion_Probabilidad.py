import numpy as np
import matplotlib.pyplot as plt

class ItemDistribution:
    def __init__(self, item_probabilities):
        self.item_probabilities = item_probabilities  # Probabilidades de elección de cada bien
        self.items = list(item_probabilities.keys())

    def simulate_choices(self, num_players):
        # Simular las elecciones de los jugadores
        choices = np.random.choice(self.items, size=num_players, p=list(self.item_probabilities.values()))
        return choices

    def plot_distribution(self, choices):
        # Graficar la distribución de elecciones
        unique, counts = np.unique(choices, return_counts=True)
        plt.bar(unique, counts, color='skyblue')
        plt.xlabel('Bienes')
        plt.ylabel('Número de Jugadores')
        plt.title('Distribución de Elección de Bienes')
        plt.show()

# Probabilidades de elección de cada bien
item_probabilities = {
    'Espada': 0.5,
    'Arco': 0.3,
    'Escudo': 0.2
}

# Crear el modelo de distribución de bienes
item_distribution = ItemDistribution(item_probabilities)

# Simular las elecciones de 1000 jugadores
num_players = 1000
choices = item_distribution.simulate_choices(num_players)

# Graficar la distribución de elecciones
item_distribution.plot_distribution(choices)

# Mostrar la distribución de elecciones
unique, counts = np.unique(choices, return_counts=True)
probability_distribution = dict(zip(unique, counts / num_players))
print("Distribución de probabilidad de elección de bienes:")
for item, probability in probability_distribution.items():
    print(f"{item}: {probability:.2f}")
