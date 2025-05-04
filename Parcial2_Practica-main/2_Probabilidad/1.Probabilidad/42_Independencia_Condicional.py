import numpy as np

class BossAttackStrength:
    def __init__(self, player_level_probs, weapon_type_probs, attack_strength_probs):
        self.player_level_probs = player_level_probs  # Probabilidades de nivel del jugador
        self.weapon_type_probs = weapon_type_probs  # Probabilidades de tipo de arma dado el nivel del jugador
        self.attack_strength_probs = attack_strength_probs  # Probabilidades de fuerza de ataque dado el tipo de arma

    def calculate_attack_strength(self, player_level):
        # Calcular la fuerza de ataque esperada del jefe dado el nivel del jugador
        expected_strength = 0
        for weapon, weapon_prob in self.weapon_type_probs[player_level].items():
            for strength, strength_prob in self.attack_strength_probs[weapon].items():
                expected_strength += weapon_prob * strength_prob * strength
        return expected_strength

# Probabilidades de nivel del jugador
player_level_probs = {
    'beginner': 0.5,
    'intermediate': 0.3,
    'advanced': 0.2
}

# Probabilidades de tipo de arma dado el nivel del jugador
weapon_type_probs = {
    'beginner': {'sword': 0.6, 'bow': 0.4},
    'intermediate': {'sword': 0.5, 'bow': 0.3, 'staff': 0.2},
    'advanced': {'sword': 0.4, 'bow': 0.2, 'staff': 0.4}
}

# Probabilidades de fuerza de ataque dado el tipo de arma
attack_strength_probs = {
    'sword': {10: 0.3, 20: 0.5, 30: 0.2},
    'bow': {15: 0.4, 25: 0.4, 35: 0.2},
    'staff': {20: 0.3, 30: 0.4, 40: 0.3}
}

# Crear el modelo de fuerza de ataque del jefe
boss_attack = BossAttackStrength(player_level_probs, weapon_type_probs, attack_strength_probs)

# Calcular la fuerza de ataque esperada del jefe para un jugador de nivel intermedio
player_level = 'intermediate'
expected_strength = boss_attack.calculate_attack_strength(player_level)
print(f"Fuerza de ataque esperada del jefe para un jugador de nivel {player_level}: {expected_strength:.2f}")
