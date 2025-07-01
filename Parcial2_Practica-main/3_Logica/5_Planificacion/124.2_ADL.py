# Implementación básica de ADL (Action Description Language) como recurso de planificación
# Este ejemplo define acciones con precondiciones y efectos, y un planificador simple que busca un plan para alcanzar un objetivo.

# Definición de una acción ADL
class Action:
    def __init__(self, name, preconditions, effects):
        self.name = name
        self.preconditions = preconditions  # Lista de condiciones que deben cumplirse antes de ejecutar la acción
        self.effects = effects              # Lista de efectos que resultan de ejecutar la acción

    def is_applicable(self, state):
        # Verifica si la acción puede aplicarse en el estado actual
        return all(cond in state for cond in self.preconditions)

    def apply(self, state):
        # Aplica los efectos de la acción al estado actual y retorna el nuevo estado
        new_state = state.copy()
        for effect in self.effects:
            if effect.startswith('not '):
                # Elimina la condición si el efecto es negativo
                cond = effect[4:]
                if cond in new_state:
                    new_state.remove(cond)
            else:
                # Agrega la condición si el efecto es positivo
                if effect not in new_state:
                    new_state.append(effect)
        return new_state

# Planificador simple que busca un plan secuencial para alcanzar el objetivo
def plan(initial_state, goal, actions, max_depth=10):
    # Búsqueda en profundidad limitada para encontrar un plan
    def dfs(state, path, depth):
        if all(g in state for g in goal):
            return path
        if depth == 0:
            return None
        for action in actions:
            if action.is_applicable(state):
                new_state = action.apply(state)
                if new_state != state:  # Evita ciclos triviales
                    result = dfs(new_state, path + [action.name], depth - 1)
                    if result is not None:
                        return result
        return None

    return dfs(initial_state, [], max_depth)

# Ejemplo de uso

# Definimos acciones en ADL
actions = [
    Action(
        name="Coger llave",
        preconditions=["en sala", "llave en sala"],
        effects=["tiene llave", "not llave en sala"]
    ),
    Action(
        name="Abrir puerta",
        preconditions=["en sala", "tiene llave", "puerta cerrada"],
        effects=["puerta abierta", "not puerta cerrada"]
    ),
    Action(
        name="Salir",
        preconditions=["en sala", "puerta abierta"],
        effects=["fuera", "not en sala"]
    )
]

# Estado inicial y objetivo
initial_state = ["en sala", "llave en sala", "puerta cerrada"]
goal = ["fuera"]

# Ejecutamos el planificador
plan_result = plan(initial_state, goal, actions)

# Mostramos el resultado
if plan_result:
    print("Plan encontrado:")
    for step in plan_result:
        print(" -", step)
else:
    print("No se encontró un plan.")

# Fin del programa
