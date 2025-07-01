from itertools import product
from pysat.solvers import Glucose3

# SATPLAN para Planificación de Recursos Ambientales
# --------------------------------------------------
# Este programa implementa el algoritmo SATPLAN para resolver problemas de planificación,
# orientado a la gestión de recursos ambientales (por ejemplo, reforestación, limpieza de ríos, etc.).
# SATPLAN convierte el problema de planificación en una fórmula SAT y utiliza un solucionador SAT para encontrar un plan.


# 1. Definición del problema de planificación ambiental
# ----------------------------------------------------
# Definimos acciones, precondiciones, efectos y estado inicial/meta.

# Ejemplo: Reforestar un área y limpiar un río

# Estados posibles
fluents = [
    'arboles_plantados',   # Hay árboles plantados
    'rio_limpio',          # El río está limpio
    'herramientas_disponibles', # Hay herramientas disponibles
]

# Acciones posibles
actions = [
    {
        'name': 'plantar_arboles',
        'pre': ['herramientas_disponibles'],
        'add': ['arboles_plantados'],
        'del': []
    },
    {
        'name': 'limpiar_rio',
        'pre': ['herramientas_disponibles'],
        'add': ['rio_limpio'],
        'del': []
    },
    {
        'name': 'conseguir_herramientas',
        'pre': [],
        'add': ['herramientas_disponibles'],
        'del': []
    }
]

# Estado inicial y meta
initial_state = {
    'arboles_plantados': False,
    'rio_limpio': False,
    'herramientas_disponibles': False
}
goal_state = {
    'arboles_plantados': True,
    'rio_limpio': True
}

# 2. Codificación de variables proposicionales
# --------------------------------------------
# Cada variable representa un fluente o acción en un tiempo t.

def varnum(name, t):
    """Genera un número único para cada variable proposicional."""
    base = {f: i+1 for i, f in enumerate(fluents)}
    base.update({a['name']: len(fluents)+i+1 for i, a in enumerate(actions)})
    return base[name] + t * (len(fluents) + len(actions))

# 3. Generación de la fórmula SAT
# -------------------------------
# Creamos las restricciones para SATPLAN: estado inicial, acciones, precondiciones, efectos, exclusiones y meta.

def encode_satplan(horizon):
    clauses = []

    # Estado inicial
    for f in fluents:
        lit = varnum(f, 0)
        if initial_state[f]:
            clauses.append([lit])
        else:
            clauses.append([-lit])

    # Restricciones por cada paso de tiempo
    for t in range(horizon):
        # Acción implica precondiciones
        for a in actions:
            a_var = varnum(a['name'], t)
            for pre in a['pre']:
                clauses.append([-a_var, varnum(pre, t)])

        # Efectos de acciones
        for a in actions:
            a_var = varnum(a['name'], t)
            for add in a['add']:
                clauses.append([-a_var, varnum(add, t+1)])
            for d in a['del']:
                clauses.append([-a_var, -varnum(d, t+1)])

        # Persistencia de fluentes (frame axioms)
        for f in fluents:
            # Si no se borra, persiste
            del_actions = [a for a in actions if f in a['del']]
            if del_actions:
                clauses.append([
                    -varnum(f, t),
                    varnum(f, t+1),
                    *[varnum(a['name'], t) for a in del_actions]
                ])
            else:
                clauses.append([-varnum(f, t), varnum(f, t+1)])

    # Meta en el último paso
    for f in goal_state:
        if goal_state[f]:
            clauses.append([varnum(f, horizon)])

    return clauses

# 4. Búsqueda de un plan usando un SAT solver
# -------------------------------------------
def extract_plan(model, horizon):
    plan = []
    for t in range(horizon):
        for a in actions:
            if varnum(a['name'], t) in model:
                plan.append((t, a['name']))
    return plan

def satplan_search(max_horizon=5):
    for h in range(1, max_horizon+1):
        clauses = encode_satplan(h)
        solver = Glucose3()
        for clause in clauses:
            solver.add_clause(clause)
        if solver.solve():
            model = solver.get_model()
            plan = extract_plan(model, h)
            print(f"Plan encontrado en {h} pasos:")
            for t, action in plan:
                print(f"  Paso {t+1}: {action}")
            return
    print("No se encontró un plan en el horizonte dado.")

# 5. Ejecución principal
# ----------------------
if __name__ == "__main__":
    satplan_search(max_horizon=5)
