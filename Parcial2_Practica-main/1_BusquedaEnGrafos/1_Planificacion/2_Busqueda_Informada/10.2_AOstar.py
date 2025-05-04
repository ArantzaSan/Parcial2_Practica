import heapq

def heuristic(node, goal):
    # Ejemplo de heurística simple: 
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def ao_star(graph, start, goal):
    # Cola de prioridad para mantener los nodos a explorar
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic(start, goal), start))

    # Diccionario para mantener el registro de los costos acumulados
    costs = {start: 0}

    # Diccionario para mantener el registro de los nodos AND/OR
    node_type = {start: 'OR'}

    while priority_queue:
        # Sacamos el nodo con el menor costo estimado
        (current_cost, node) = heapq.heappop(priority_queue)

        # Si alcanzamos el nodo objetivo, devolvemos el costo acumulado
        if node == goal:
            return costs[node]

        # Exploramos los vecinos del nodo actual
        if node_type[node] == 'OR':
            for neighbor, cost, n_type in graph[node]:
                new_cost = costs[node] + cost
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(priority_queue, (priority, neighbor))
                    node_type[neighbor] = n_type

        elif node_type[node] == 'AND':
            and_cost = 0
            for neighbor, cost, n_type in graph[node]:
                new_cost = costs[node] + cost
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    and_cost += new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(priority_queue, (priority, neighbor))
                    node_type[neighbor] = n_type
            costs[node] = and_cost

    # Si no encontramos el nodo objetivo, devolvemos None
    return None

# Ejemplo de grafo AND/OR representado como un diccionario de listas de tuplas (nodo, costo, tipo)
graph = {
    (0, 0): [((1, 0), 1, 'OR'), ((0, 1), 1, 'AND')],
    (1, 0): [((1, 1), 1, 'OR')],
    (0, 1): [((1, 1), 1, 'OR')],
    (1, 1): [((1, 2), 1, 'OR')],
    (1, 2): []
}

# Ejecutamos la búsqueda AO* desde el nodo (0, 0) hasta el nodo (1, 2)
cost = ao_star(graph, (0, 0), (1, 2))
print(f"Costo mínimo desde (0, 0) hasta (1, 2) usando AO*: {cost}")
