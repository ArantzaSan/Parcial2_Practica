import random
import time

# 132_Continual_Multiagente.py
# Ejemplo básico de planificación continual multiagente en Python


# Definición del entorno compartido
class Entorno:
    def __init__(self):
        self.estado = {"recursos": 10}  # Estado global simple

    def consumir_recurso(self, cantidad):
        if self.estado["recursos"] >= cantidad:
            self.estado["recursos"] -= cantidad
            return True
        return False

    def reponer_recurso(self, cantidad):
        self.estado["recursos"] += cantidad

# Definición del agente
class Agente:
    def __init__(self, nombre, entorno):
        self.nombre = nombre
        self.entorno = entorno
        self.plan = []

    # Genera un plan simple basado en el estado actual
    def planificar(self):
        if self.entorno.estado["recursos"] > 0:
            self.plan = ["consumir"]
        else:
            self.plan = ["esperar"]

    # Ejecuta el siguiente paso del plan
    def ejecutar(self):
        if not self.plan:
            self.planificar()
        accion = self.plan.pop(0)
        if accion == "consumir":
            exito = self.entorno.consumir_recurso(1)
            if exito:
                print(f"{self.nombre} consumió 1 recurso.")
            else:
                print(f"{self.nombre} no pudo consumir recurso.")
        elif accion == "esperar":
            print(f"{self.nombre} espera recursos.")

    # Planificación continual: revisa y ajusta el plan en cada ciclo
    def ciclo(self):
        self.planificar()
        self.ejecutar()

# Configuración de la simulación multiagente
def simulacion():
    entorno = Entorno()
    agentes = [Agente(f"Agente_{i+1}", entorno) for i in range(3)]

    # Simulación de ciclos de planificación continual
    for ciclo in range(10):
        print(f"\n--- Ciclo {ciclo+1} ---")
        for agente in agentes:
            agente.ciclo()
        # Ocasionalmente reponer recursos
        if random.random() < 0.3:
            entorno.reponer_recurso(2)
            print("Se repusieron 2 recursos en el entorno.")
        time.sleep(0.5)  # Pausa para visualizar la simulación

if __name__ == "__main__":
    simulacion()
