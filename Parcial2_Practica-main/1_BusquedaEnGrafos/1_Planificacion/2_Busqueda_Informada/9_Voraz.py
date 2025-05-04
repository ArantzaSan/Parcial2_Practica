import heapq

def heuristic(node, goal):
    # Ejemplo de heurística:
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def greedy_best_first_search(graph, start, goal):
    # Cola de prioridad para mantener los nodos a explorar
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic(start, goal), start, [start]))

    # Conjunto para mantener el registro de los nodos visitados
    visited = set()

    while priority_queue:
        # Sacamos el nodo con la menor heurística
        (_, node, path) = heapq.heappop(priority_queue)

        # Si alcanzamos el nodo objetivo, devolvemos el camino
        if node == goal:
            return path

        # Si el nodo no ha sido visitado
        if node not in visited:
            visited.add(node)

            # Exploramos los vecinos del nodo actual
            for neighbor, _ in graph[node]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    priority = heuristic(neighbor, goal)
                    heapq.heappush(priority_queue, (priority, neighbor, new_path))

    # Si no encontramos el nodo objetivo, devolvemos None
    return None

# Ejemplo de grafo representado como un diccionario de listas de tuplas (nodo, costo)
graph = {
    (0, 0): [((1, 0), 1), ((0, 1), 1)],
    (1, 0): [((0, 0), 1), ((1, 1), 1)],
    (0, 1): [((0, 0), 1), ((1, 1), 1)],
    (1, 1): [((1, 0), 1), ((0, 1), 1), ((1, 2), 1)],
    (1, 2): [((1, 1), 1)]
}

# Ejecutamos la búsqueda voraz desde el nodo (0, 0) hasta el nodo (1, 2)
path = greedy_best_first_search(graph, (0, 0), (1, 2))
print(f"Camino desde (0, 0) hasta (1, 2) usando Búsqueda Voraz: {path}")
