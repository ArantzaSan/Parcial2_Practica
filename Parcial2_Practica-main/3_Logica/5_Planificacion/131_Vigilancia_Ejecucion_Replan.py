import random
import time

# 131_Vigilancia_Ejecucion_Replan.py

"""
Este programa implementa un ciclo básico de planificación con vigilancia, ejecución y replanificación.
Se simula un agente que sigue un plan, verifica si el plan sigue siendo válido (vigilancia),
ejecuta acciones y replanifica si ocurre un fallo.
"""


# Definición de un plan simple como una lista de acciones
plan_inicial = ["moverse_a", "recoger_objeto", "entregar_objeto"]

def vigilancia(plan):
    """
    Verifica si el plan sigue siendo válido.
    Simula la detección de fallos aleatorios en el entorno.
    """
    # Simulación: 20% de probabilidad de que el plan falle
    if random.random() < 0.2:
        print("Vigilancia: ¡Fallo detectado en el plan!")
        return False
    print("Vigilancia: El plan sigue siendo válido.")
    return True

def ejecucion(plan):
    """
    Ejecuta el plan acción por acción.
    Si ocurre un fallo durante la ejecución, se detiene.
    """
    for accion in plan:
        print(f"Ejecución: Realizando acción '{accion}'...")
        time.sleep(1)  # Simula el tiempo de ejecución
        # Simulación: 10% de probabilidad de fallo en cada acción
        if random.random() < 0.1:
            print(f"Ejecución: ¡Fallo durante la acción '{accion}'!")
            return False
    print("Ejecución: Plan ejecutado con éxito.")
    return True

def replan():
    """
    Genera un nuevo plan en caso de fallo.
    Aquí simplemente se devuelve un plan alternativo.
    """
    print("Replanificación: Generando un nuevo plan...")
    nuevo_plan = ["moverse_a", "pedir_ayuda", "recoger_objeto", "entregar_objeto"]
    print(f"Replanificación: Nuevo plan generado: {nuevo_plan}")
    return nuevo_plan

def ciclo_planificacion(plan):
    """
    Ciclo principal: vigilancia -> ejecución -> replanificación si es necesario.
    """
    while True:
        if not vigilancia(plan):
            plan = replan()
            continue
        if not ejecucion(plan):
            plan = replan()
            continue
        print("Ciclo completado exitosamente.")
        break

if __name__ == "__main__":
    # Inicio del ciclo de planificación
    ciclo_planificacion(plan_inicial)
