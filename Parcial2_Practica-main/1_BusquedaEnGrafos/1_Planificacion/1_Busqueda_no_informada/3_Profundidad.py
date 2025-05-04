def dfs(graph, start, goal):
    # Creamos una fila para mantener el orden de los nodos a visitar
    stack = [(start, [start])]
    # Creamos un conjunto para mantener el registro de los nodos visitados
    visited = set()

    while stack:
        # Sacamos el nodo de la parte superior de la fila
        (node, path) = stack.pop()

        # Si alcanzamos el nodo objetivo, devolvemos el camino
        if node == goal:
            return path

        # Si el nodo no ha sido visitado
        if node not in visited:
            visited.add(node)

            # Agregamos todos los vecinos no visitados a la pila
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

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

# Ejecutamos la búsqueda en profundidad desde el nodo 'A' hasta el nodo 'F'
path = dfs(graph, 'A', 'F')
print(f"Camino desde 'A' hasta 'F': {path}")
