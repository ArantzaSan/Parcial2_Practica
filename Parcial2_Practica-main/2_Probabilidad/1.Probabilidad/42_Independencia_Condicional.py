import numpy as np

class FuerzaAtaqueJefe:
    def __init__(self, probs_nivel_jugador, probs_tipo_arma, probs_fuerza_ataque):
        self.probs_nivel_jugador = probs_nivel_jugador  # Probabilidades de nivel del jugador
        self.probs_tipo_arma = probs_tipo_arma  # Probabilidades de tipo de arma dado el nivel del jugador
        self.probs_fuerza_ataque = probs_fuerza_ataque  # Probabilidades de fuerza de ataque dado el tipo de arma

    def calcular_fuerza_esperada(self, nivel_jugador):
        # Calcular la fuerza de ataque esperada del jefe dado el nivel del jugador
        fuerza_esperada = 0
        for arma, prob_arma in self.probs_tipo_arma[nivel_jugador].items():
            for fuerza, prob_fuerza in self.probs_fuerza_ataque[arma].items():
                fuerza_esperada += prob_arma * prob_fuerza * fuerza
        return fuerza_esperada

# Probabilidades de nivel del jugador
probs_nivel = {
    'principiante': 0.5,
    'intermedio': 0.3,
    'avanzado': 0.2
}

# Probabilidades de tipo de arma dado el nivel del jugador
probs_arma_por_nivel = {
    'principiante': {'espada': 0.6, 'arco': 0.4},
    'intermedio': {'espada': 0.5, 'arco': 0.3, 'bastón': 0.2},
    'avanzado': {'espada': 0.4, 'arco': 0.2, 'bastón': 0.4}
}

# Probabilidades de fuerza de ataque dado el tipo de arma
probs_fuerza_por_arma = {
    'espada': {10: 0.3, 20: 0.5, 30: 0.2},
    'arco': {15: 0.4, 25: 0.4, 35: 0.2},
    'bastón': {20: 0.3, 30: 0.4, 40: 0.3}
}

# Crear el modelo de fuerza de ataque del jefe
jefe_ataque = FuerzaAtaqueJefe(probs_nivel, probs_arma_por_nivel, probs_fuerza_por_arma)

# Calcular la fuerza de ataque esperada del jefe para un jugador de nivel intermedio
nivel_jugador = 'intermedio'
fuerza_esperada = jefe_ataque.calcular_fuerza_esperada(nivel_jugador)
print(f"Fuerza de ataque esperada del jefe para un jugador de nivel {nivel_jugador}: {fuerza_esperada:.2f}")
