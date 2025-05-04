from itertools import product

def solve_cutset_conditioning(variables, domains, constraints, cutset):
    # Generar todas las combinaciones posibles para el cutset
    cutset_assignments = list(product(*[domains[var] for var in cutset]))

    for assignment in cutset_assignments:
        cutset_solution = dict(zip(cutset, assignment))

        # Verificar si la asignación del cutset es consistente con las restricciones
        if all(constraint(cutset_solution) for constraint in constraints):
            # Resolver el resto del problema con la asignación del cutset fija
            remaining_solution = backtrack(variables, domains, constraints, cutset_solution)
            if remaining_solution:
                return {**cutset_solution, **remaining_solution}

    return None

def backtrack(variables, domains, constraints, assignment):
    if len(assignment) == len(variables):
        return assignment

    unassigned = [var for var in variables if var not in assignment]
    var = unassigned[0]

    for value in domains[var]:
        new_assignment = {**assignment, var: value}
        if all(constraint(new_assignment) for constraint in constraints):
            result = backtrack(variables, domains, constraints, new_assignment)
            if result:
                return result

    return None

# Definir variables, dominios y restricciones para el sistema de ventas de automóviles
variables = ['auto1', 'auto2', 'auto3']
domains = {
    'auto1': ['rojo', 'azul'],
    'auto2': ['azul', 'verde'],
    'auto3': ['rojo', 'verde']
}

# Restricciones: no dos autos adyacentes pueden tener el mismo color
def constraint(assignment):
    values = list(assignment.values())
    return len(values) == len(set(values))

constraints = [constraint]

# Definir el cutset
cutset = ['auto1']

# Resolver el problema utilizando acondicionamiento de corte
solution = solve_cutset_conditioning(variables, domains, constraints, cutset)

if solution:
    print(f"Solución encontrada: {solution}")
else:
    print("No se encontró solución.")
