import numpy as np
from scipy.optimize import linprog

class RockPaperScissorsGame:
    def __init__(self):
        # Matriz de pagos para el jugador 1 (filas) contra el jugador 2 (columnas)
        # 1 = ganar, 0 = empatar, -1 = perder
        self.payoff_matrix = np.array([
            [0, -1, 1],   # Piedra
            [1, 0, -1],   # Papel
            [-1, 1, 0]    # Tijera
        ])

    def find_nash_equilibrium(self):
        # Resolver el problema de programación lineal para encontrar la estrategia óptima
        n = len(self.payoff_matrix)
        c = np.zeros(n + 1)
        c[-1] = -1

        A_ub = np.zeros((n, n + 1))
        for i in range(n):
            A_ub[i, :n] = -self.payoff_matrix[i]
            A_ub[i, n] = 1

        b_ub = np.zeros(n)
        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0
        b_eq = np.ones(1)

        bounds = [(0, None) for _ in range(n + 1)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            strategy = result.x[:-1]
            return strategy / np.sum(strategy)
        else:
            raise ValueError("No se pudo encontrar el equilibrio de Nash.")

    def play_game(self, player_strategy, opponent_strategy):
        # Simular un juego y calcular el pago esperado
        payoff = np.dot(player_strategy, np.dot(self.payoff_matrix, opponent_strategy))
        return payoff

# Crear el juego y encontrar la estrategia óptima
game = RockPaperScissorsGame()
optimal_strategy = game.find_nash_equilibrium()
print("Estrategia óptima (equilibrio de Nash):")
print(f"Piedra: {optimal_strategy[0]:.2f}, Papel: {optimal_strategy[1]:.2f}, Tijera: {optimal_strategy[2]:.2f}")

# Simular un juego contra un oponente con una estrategia fija
opponent_strategy = np.array([0.4, 0.3, 0.3])  # Ejemplo: oponente juega piedra 40%, papel 30%, tijera 30%
payoff = game.play_game(optimal_strategy, opponent_strategy)
print(f"Pago esperado contra el oponente: {payoff:.2f}")
