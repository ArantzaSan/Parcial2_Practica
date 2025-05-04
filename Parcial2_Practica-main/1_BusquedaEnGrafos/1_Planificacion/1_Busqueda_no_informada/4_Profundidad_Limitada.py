def dfs_limited(graph, start, goal, depth_limit):
    # Creamos una fila para mantener el orden de los nodos a visitar
    stack = [(start, [start], 0)]
    # Creamos un conjunto para mantener el registro de los nodos visitados
    visited = set()

    while stack:
        # Sacamos el nodo de la parte superior de la fila
        (node, path, depth) = stack.pop()

        # Si alcanzamos el nodo objetivo, devolvemos el camino
        if node == goal:
            return path

        # Si el nodo no ha sido visitado y no hemos alcanzado el límite de profundidad
        if node not in visited and depth < depth_limit:
            visited.add(node)

            # Agregamos todos los vecinos no visitados a la fila
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))

    # Si no encontramos el nodo objetivo, devolvemos None
    return None

# Ejemplo de grafo representado como un diccionario de listas
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutamos la búsqueda en profundidad limitada desde el nodo 'A' hasta el nodo 'F' con un límite de profundidad de 3
path = dfs_limited(graph, 'A', 'F', 3)
print(f"Camino desde 'A' hasta 'F' con profundidad limitada: {path}")
