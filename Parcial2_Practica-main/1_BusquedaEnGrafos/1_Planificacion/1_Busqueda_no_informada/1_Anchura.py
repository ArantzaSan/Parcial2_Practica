from collections import deque

def bfs(graph, start):
    queue = deque([start]) #Se crea una lista de espera
    visited = set()

while queue:
    node = queue.popleft() #Se determina un nodo de la lista de espera

    # Si el nodo no fue visitado, se visita
    if node not in visited:
        print("Visitando nodo: {node}")
        visited.add(node) #Se marca su visita

        for neighbor in graph[node]: #Se agenda a visitar sus vecinos
            if neighbor not in vidited:
                queue.append(neighbor)
# lista para ejemplificar el grafo
graph = {
    'A': ['B','C'],
    'B': ['A','D','E'],
    'C': ['A','F'],
    'D': ['B'],
    'E': ['B','F'],
    'F': ['C','E']
}
# Se ejecuta la busqueda con el nodo A
bfs(graph, 'A')