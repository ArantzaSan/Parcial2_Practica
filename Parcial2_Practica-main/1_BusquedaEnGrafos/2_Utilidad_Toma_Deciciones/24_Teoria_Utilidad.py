import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo dirigido
G = nx.DiGraph()

# Definir los nodos y las aristas con sus utilidades
edges = [
    ('A', 'B', 4),
    ('A', 'C', 2),
    ('B', 'D', 5),
    ('B', 'E', 1),
    ('C', 'F', 3),
    ('C', 'G', 2),
    ('D', 'H', 2),
    ('E', 'H', 3),
    ('F', 'H', 2),
    ('G', 'H', 2)
]

# Añadir las aristas al grafo
for start, end, utility in edges:
    G.add_edge(start, end, utility=utility)

# Función para encontrar el camino con mayor utilidad
def max_utility_path(graph, start, goal):
    # Usar una variación del algoritmo de Dijkstra para maximizar la utilidad
    paths = {node: {'path': [], 'utility': 0} for node in graph.nodes}
    paths[start]['utility'] = 0

    unvisited = set(graph.nodes)

    while unvisited:
        # Encontrar el nodo con la mayor utilidad acumulada
        current = max((node for node in unvisited if node in paths), key=lambda n: paths[n]['utility'])
        unvisited.remove(current)

        if current == goal:
            break

        for neighbor in graph.neighbors(current):
            if neighbor in unvisited:
                new_utility = paths[current]['utility'] + graph[current][neighbor]['utility']
                if new_utility > paths[neighbor]['utility']:
                    paths[neighbor] = {'path': paths[current]['path'] + [current], 'utility': new_utility}

    return paths[goal]['path'] + [goal], paths[goal]['utility']

# Encontrar el camino con mayor utilidad desde el nodo 'A' hasta el nodo 'H'
path, utility = max_utility_path(G, 'A', 'H')
print(f"Camino con mayor utilidad: {path}")
print(f"Utilidad total: {utility}")

# Dibujar el grafo
pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'utility')
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Grafo de Decisiones con Utilidades")
plt.show()
