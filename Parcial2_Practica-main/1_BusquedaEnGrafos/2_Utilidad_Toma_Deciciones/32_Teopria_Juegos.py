import numpy as np
from scipy.optimize import linprog

class JuegoPiedraPapelTijera:
    def __init__(self):
        # Matriz de pagos para el Jugador 1 (filas) contra el Jugador 2 (columnas)
        # 1 = ganar, 0 = empatar, -1 = perder
        self.matriz_pagos = np.array([
            [0, -1, 1],    # Piedra
            [1, 0, -1],    # Papel
            [-1, 1, 0]     # Tijera
        ])

    def encontrar_equilibrio_nash(self):
        # Resolver el problema de programación lineal para encontrar la estrategia óptima
        n = len(self.matriz_pagos)
        c = np.zeros(n + 1)
        c[-1] = -1

        A_ub = np.zeros((n, n + 1))
        for i in range(n):
            A_ub[i, :n] = -self.matriz_pagos[i]
            A_ub[i, n] = 1

        b_ub = np.zeros(n)
        A_eq = np.ones((1, n + 1))
        A_eq[0, -1] = 0
        b_eq = np.ones(1)

        limites = [(0, None) for _ in range(n + 1)]

        resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=limites, method='highs')

        if resultado.success:
            estrategia = resultado.x[:-1]
            return estrategia / np.sum(estrategia)
        else:
            raise ValueError("No se pudo encontrar el equilibrio de Nash.")

    def jugar_juego(self, estrategia_jugador, estrategia_oponente):
        # Simular un juego y calcular el pago esperado
        pago = np.dot(estrategia_jugador, np.dot(self.matriz_pagos, estrategia_oponente))
        return pago

# Crear el juego y encontrar la estrategia óptima
juego = JuegoPiedraPapelTijera()
estrategia_optima = juego.encontrar_equilibrio_nash()
print("Estrategia óptima (equilibrio de Nash):")
print(f"Piedra: {estrategia_optima[0]:.2f}, Papel: {estrategia_optima[1]:.2f}, Tijera: {estrategia_optima[2]:.2f}")

# Simular un juego contra un oponente con una estrategia fija
estrategia_rival = np.array([0.4, 0.3, 0.3])  # Ejemplo: oponente juega piedra 40%, papel 30%, tijera 30%
pago_esperado = juego.jugar_juego(estrategia_optima, estrategia_rival)
print(f"Pago esperado contra el oponente: {pago_esperado:.2f}")
