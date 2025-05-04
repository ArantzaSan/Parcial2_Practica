from collections import defaultdict

# Se define la clase grafo con vertices
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

# detecta restriccion segura
def is_safe(graph, vertex, color, colors):
    for neighbor in graph.graph[vertex]:
        if colors[neighbor] == color:
            return False
    return True

# Coloreo del grafo
def graph_coloring(graph, m, vertex=0):
    if vertex == graph.vertices:
        return True

    for color in range(1, m + 1):
        if is_safe(graph, vertex, color, colors):
            colors[vertex] = color
            if graph_coloring(graph, m, vertex + 1):
                return True
            colors[vertex] = 0

    return False

def solve_csp(graph, m):
    colors = [0] * graph.vertices
    if graph_coloring(graph, m):
        return colors
    else:
        return None

# Ejemplo de grafo
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(1, 3)

# Número de colores disponibles
m = 3

# Resolver el problema de coloreado de grafos
solution = solve_csp(g, m)

if solution:
    print(f"Solución encontrada: {solution}")
else:
    print("No se encontró solución.")
