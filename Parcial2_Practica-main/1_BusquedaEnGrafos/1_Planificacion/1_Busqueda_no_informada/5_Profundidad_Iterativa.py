def dfs_limited(graph, node, goal, depth_limit, path):
    # Si alcanzamos el nodo objetivo, devolvemos el camino
    if node == goal:
        return path

    # Si hemos alcanzado el límite de profundidad, devolvemos None
    if depth_limit <= 0:
        return None

    # Exploramos los vecinos del nodo actual
    for neighbor in graph[node]:
        if neighbor not in path:  # Evitamos ciclos
            new_path = dfs_limited(graph, neighbor, goal, depth_limit - 1, path + [neighbor])
            if new_path:
                return new_path

    return None

def iddfs(graph, start, goal):
    depth_limit = 0
    while True:
        # Realizamos una búsqueda en profundidad limitada con el límite de profundidad actual
        path = dfs_limited(graph, start, goal, depth_limit, [start])
        if path:
            return path
        # Incrementamos el límite de profundidad
        depth_limit += 1

# Ejemplo de grafo representado como un diccionario de listas
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutamos la búsqueda en profundidad iterativa desde el nodo 'A' hasta el nodo 'F'
path = iddfs(graph, 'A', 'F')
print(f"Camino desde 'A' hasta 'F' con profundidad iterativa: {path}")
