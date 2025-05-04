from collections import defaultdict

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.neighbors = self.build_neighbors()

    def build_neighbors(self):
        neighbors = defaultdict(set)
        for (x, y) in self.constraints:
            neighbors[x].add(y)
            neighbors[y].add(x)
        return neighbors

    def is_consistent(self, var, value, assignment):
        for neighbor in self.neighbors[var]:
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

    def ac3(self):
        queue = [(x, y) for (x, y) in self.constraints]
        while queue:
            (xi, xj) = queue.pop(0)
            if self.revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                for xk in self.neighbors[xi]:
                    if xk != xi:
                        queue.append((xk, xi))
        return True

    def revise(self, xi, xj):
        revised = False
        to_remove = set()
        for x in self.domains[xi]:
            if all(not self.is_consistent(xj, y, {xi: x}) for y in self.domains[xj]):
                to_remove.add(x)
                revised = True
        self.domains[xi] -= to_remove
        return revised

def solve_csp(variables, domains, constraints):
    csp = CSP(variables, domains, constraints)
    if csp.ac3():
        return csp.domains
    else:
        return None

# Ejemplo de problema de asignación de colores
variables = ['A', 'B', 'C']
domains = {
    'A': {'rojo', 'verde', 'azul'},
    'B': {'rojo', 'verde', 'azul'},
    'C': {'rojo', 'verde', 'azul'}
}
constraints = [
    ('A', 'B'),
    ('B', 'C')
]

# Resolver el problema de asignación de colores
solution = solve_csp(variables, domains, constraints)

if solution:
    print(f"Solución encontrada: {solution}")
else:
    print("No se encontró solución.")
