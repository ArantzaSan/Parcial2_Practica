import heapq

def uniform_cost_search(graph, start, goal):
    # Se crea una fila prority a visitar
    priority_queue = [(0, start)] 
    costs = {start: 0} #Directorio registro costos
    visited = set() #Directorio de registro de nodos visitados

    while priority_queue:
        # Nodo con el costo más bajo
        current_cost, node = heapq.heappop(priority_queue)

        # Se alcanza el nodo objetivo, devuelve costo acumulado
        if node == goal:
            return current_cost

        # Si el nodo no ha sido visitado
        if node not in visited:
            visited.add(node)

            # Explora los vecinos del nodo actual
            for neighbor, cost in graph[node]:
                new_cost = current_cost + cost
                # Si encontramos un camino cercano para llegar a este vecino
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    # Se agrega el vecino a la cola de priority con nueva cercania
                    heapq.heappush(priority_queue, (new_cost, neighbor))

    # Si no encontramos el nodo objetivo, devuelve infinito
    return float('inf')

# Ejemplo de grafo como un diccionario de listas de tuplas (nodo, costo)
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('D', 2), ('E', 5)],
    'C': [('A', 4), ('F', 2)],
    'D': [('B', 2)],
    'E': [('B', 5), ('F', 1)],
    'F': [('C', 2), ('E', 1)]
}

# Ejecutamos la búsqueda de costo uniforme desde el nodo 'A' hasta el nodo 'F'
cost = uniform_cost_search(graph, 'A', 'F')
print(f"Costo mínimo desde 'A' hasta 'F': {cost}")
