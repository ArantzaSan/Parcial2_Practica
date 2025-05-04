from collections import deque

# busqueda bidireccional
def bidirectional_search(graph, start, goal):
    if start == goal:
        return [start]

    # Inicializamos las colas para ambas búsquedas
    forward_queue = deque([(start, [start])])
    backward_queue = deque([(goal, [goal])])

    # Conjuntos para mantener el registro de los nodos visitados en ambas direcciones
    forward_visited = {start: [start]}
    backward_visited = {goal: [goal]}

    while forward_queue and backward_queue:
        # Expandimos un nivel en la búsqueda hacia adelante
        forward_path = expand_level(graph, forward_queue, forward_visited, backward_visited)
        if forward_path:
            return forward_path

        # Expandimos un nivel en la búsqueda hacia atrás
        backward_path = expand_level(graph, backward_queue, backward_visited, forward_visited)
        if backward_path:
            return backward_path[::-1]  # Invertimos el camino para que vaya de start a goal

    return None

# iteracion para buscar vecinos
def expand_level(graph, queue, visited, other_visited):
    for _ in range(len(queue)):
        node, path = queue.popleft()

        for neighbor in graph[node]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                visited[neighbor] = new_path
                queue.append((neighbor, new_path))

                # Verificamos si encontramos una intersección con la otra búsqueda
                if neighbor in other_visited:
                    return path + other_visited[neighbor][::-1]

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

# Ejecutamos la búsqueda bidireccional desde el nodo 'A' hasta el nodo 'F'
path = bidirectional_search(graph, 'A', 'F')
print(f"Camino desde 'A' hasta 'F' con búsqueda bidireccional: {path}")
