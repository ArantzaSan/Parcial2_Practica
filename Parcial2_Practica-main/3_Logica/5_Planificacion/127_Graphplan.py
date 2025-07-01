"""
127_Graphplan.py

Implementación básica de Graphplan orientada a un escenario de ventas.
Graphplan es un algoritmo de planificación automática que construye un grafo de planificación para encontrar un plan de acciones que lleve de un estado inicial a un objetivo.

Este ejemplo simula un proceso de ventas: contactar clientes, hacer demostraciones y cerrar ventas.

Autor: GitHub Copilot
"""

# Definición de acciones posibles en el dominio de ventas
class Action:
    def __init__(self, name, preconds, effects):
        self.name = name
        self.preconds = set(preconds)
        self.effects = set(effects)

    def __repr__(self):
        return f"Action({self.name})"

# Definición del grafo de planificación.
class PlanningGraph:
    def __init__(self, initial_state, actions, goals):
        self.levels = []  # Lista de niveles de estados
        self.actions = actions
        self.initial_state = set(initial_state)
        self.goals = set(goals)

    def expand_graph(self):
        # Expande el grafo hasta que los objetivos estén presentes o no haya cambios
        current_level = self.initial_state
        self.levels.append(current_level)
        while True:
            next_level = set(current_level)
            applicable_actions = []
            for action in self.actions:
                if action.preconds.issubset(current_level):
                    next_level.update(action.effects)
                    applicable_actions.append(action)
            if self.goals.issubset(next_level):
                self.levels.append(next_level)
                return True
            if next_level == current_level:
                return False  # No se puede avanzar más
            self.levels.append(next_level)
            current_level = next_level

    def extract_plan(self):
        # Extrae un plan hacia atrás desde el último nivel
        plan = []
        goals = self.goals.copy()
        for level in reversed(range(1, len(self.levels))):
            actions_in_level = []
            for action in self.actions:
                if action.effects & goals and action.preconds.issubset(self.levels[level-1]):
                    actions_in_level.append(action)
            if not actions_in_level:
                continue
            plan.append(actions_in_level)
            # Actualiza los objetivos para el nivel anterior
            new_goals = set()
            for action in actions_in_level:
                new_goals.update(action.preconds)
            goals = new_goals
        plan.reverse()
        return plan

# Definición del dominio de ventas
actions = [
    Action("Contactar Cliente", ["cliente_identificado"], ["cliente_contactado"]),
    Action("Hacer Demostración", ["cliente_contactado"], ["demostracion_realizada"]),
    Action("Enviar Cotización", ["demostracion_realizada"], ["cotizacion_enviada"]),
    Action("Cerrar Venta", ["cotizacion_enviada"], ["venta_cerrada"]),
]

# Estado inicial y objetivos
initial_state = ["cliente_identificado"]
goals = ["venta_cerrada"]

# Ejecución del algoritmo Graphplan
if __name__ == "__main__":
    print("Inicializando Graphplan para ventas...")
    graph = PlanningGraph(initial_state, actions, goals)
    if graph.expand_graph():
        print("¡Plan encontrado!\n")
        plan = graph.extract_plan()
        for i, actions_in_level in enumerate(plan):
            print(f"Nivel {i+1}:")
            for action in actions_in_level:
                print(f"  - {action.name}")
    else:
        print("No se pudo encontrar un plan para alcanzar los objetivos.")
