import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo dirigido
grafo_decision = nx.DiGraph()

# Definir los nodos y las aristas con sus ganancias
conexiones = [
    ('A', 'B', {'ganancia': 4}),
    ('A', 'C', {'ganancia': 2}),
    ('B', 'D', {'ganancia': 5}),
    ('B', 'E', {'ganancia': 1}),
    ('C', 'F', {'ganancia': 3}),
    ('C', 'G', {'ganancia': 2}),
    ('D', 'H', {'ganancia': 2}),
    ('E', 'H', {'ganancia': 3}),
    ('F', 'H', {'ganancia': 2}),
    ('G', 'H', {'ganancia': 2})
]

# Añadir las aristas al grafo
grafo_decision.add_edges_from([(inicio, fin, atributos) for inicio, fin, atributos in conexiones])

# Función para encontrar el camino con mayor ganancia
def camino_maxima_ganancia(grafo, inicio, objetivo):
    # Inicializar las distancias (negativas para maximizar) y los predecesores
    distancias = {nodo: float('-inf') for nodo in grafo.nodes}
    predecesores = {nodo: None for nodo in grafo.nodes}
    distancias[inicio] = 0

    nodos_prioritarios = [(0, inicio)]  # Cola de prioridad con ganancias negativas

    while nodos_prioritarios:
        ganancia_actual, nodo_actual = heapq.heappop(nodos_prioritarios)
        ganancia_actual = -ganancia_actual  # Revertir la negatividad

        if ganancia_actual > distancias[nodo_actual]:
            continue

        for vecino in grafo.successors(nodo_actual):
            ganancia_vecino = grafo[nodo_actual][vecino]['ganancia']
            nueva_ganancia = ganancia_actual + ganancia_vecino
            if nueva_ganancia > distancias[vecino]:
                distancias[vecino] = nueva_ganancia
                predecesores[vecino] = nodo_actual
