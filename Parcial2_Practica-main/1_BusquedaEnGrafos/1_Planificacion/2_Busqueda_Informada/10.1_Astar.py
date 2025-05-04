import heapq

def heuristic(node, goal):
    # Heurística: distancia Manhattan multiplicada por un factor
    return (abs(node[0] - goal[0]) + abs(node[1] - goal[1])) * 1.5

def a_star_search(graph, start, goal):
    # Estructura de datos para la frontera: (f_score, nodo, camino)
    open_set = [(0, start, [start])]
    # Diccionario para almacenar el costo g(n) conocido hasta cada nodo
    g_score = {start: 0}
    # Diccionario para recordar el nodo padre de cada nodo en el camino óptimo
    came_from = {}

    while open_set:
        # Obtener el nodo con el menor f_score de la frontera
        f_score, current, path = heapq.heappop(open_set)

        # Si el nodo actual es el objetivo, reconstruir y retornar el camino
        if current == goal:
            return path

        # Explorar los vecinos del nodo actual
        for neighbor, cost in graph[current]:
            tentative_g_score = g_score[current] + cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # Se descubre un mejor camino hacia el vecino
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                h_score = heuristic(neighbor, goal)
                f_score_neighbor = tentative_g_score + h_score
                heapq.heappush(open_set, (f_score_neighbor, neighbor, path + [neighbor]))

    # Si la frontera se vacía sin encontrar el objetivo
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
