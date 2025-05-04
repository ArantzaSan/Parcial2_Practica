import random

class RobotSurvival:
    def __init__(self, energy, location_risk, location_reward):
        self.energy = energy
        self.location_risk = location_risk  # Probabilidad de encontrar peligro en una nueva ubicación
        self.location_reward = location_reward  # Recompensa esperada en una nueva ubicación

    def calculate_expected_utility(self, action):
        if action == "move":
            # Utilidad esperada de moverse a una nueva ubicación
            expected_utility = (self.location_reward * (1 - self.location_risk)) - (self.energy * self.location_risk)
        elif action == "stay":
            # Utilidad esperada de quedarse y recargar energía
            expected_utility = self.energy * 0.5  # Supongamos que recargar da un 50% de la energía actual como utilidad
        return expected_utility

    def value_of_information(self):
        # Calcular la utilidad esperada de ambas acciones
        utility_move = self.calculate_expected_utility("move")
        utility_stay = self.calculate_expected_utility("stay")

        # Valor de la información es la diferencia entre las utilidades esperadas
        voi = abs(utility_move - utility_stay)
        return voi

    def decide_action(self):
        voi = self.value_of_information()
        print(f"Valor de la Información: {voi}")

        if voi > 10:  # Umbral arbitrario para decidir si explorar más
            print("El valor de la información es alto. El robot debería explorar más antes de decidir.")
        else:
            utility_move = self.calculate_expected_utility("move")
            utility_stay = self.calculate_expected_utility("stay")
            if utility_move > utility_stay:
                print("El robot debería moverse a una nueva ubicación.")
            else:
                print("El robot debería quedarse y recargar energía.")

# Parámetros del robot
energy = 50
location_risk = 0.3  # 30% de probabilidad de encontrar peligro
location_reward = 60  # Recompensa esperada en una nueva ubicación

# Crear el robot y tomar una decisión
robot = RobotSurvival(energy, location_risk, location_reward)
robot.decide_action()
