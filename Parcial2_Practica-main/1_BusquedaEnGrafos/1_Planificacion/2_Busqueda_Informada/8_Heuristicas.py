import heapq

def heuristic(node, goal):
    # heurística general    
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def a_star_search(graph, start, goal):
    # Cola de prioridad para mantener los nodos a explorar
    priority_queue = []
    heapq.heappush(priority_queue, (0, start, [start]))

    # Diccionario para mantener el registro de los costos acumulados
    costs = {start: 0}

    while priority_queue:
        # Sacamos el nodo con el menor costo estimado
        (current_cost, node, path) = heapq.heappop(priority_queue)

        # Si alcanzamos el nodo objetivo, devolvemos el camino
        if node == goal:
            return path

        # Exploramos los vecinos del nodo actual
        for neighbor, cost in graph[node]:
            new_cost = costs[node] + cost
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                new_path = path + [neighbor]
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

# Ejecutamos la búsqueda A* desde el nodo (0, 0) hasta el nodo (1, 2)
path = a_star_search(graph, (0, 0), (1, 2))
print(f"Camino desde (0, 0) hasta (1, 2) usando A*: {path}")
