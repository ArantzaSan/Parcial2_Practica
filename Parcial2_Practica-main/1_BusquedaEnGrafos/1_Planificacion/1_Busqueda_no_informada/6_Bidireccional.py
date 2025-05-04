from collections import deque

def bidirectional_search(graph, start, goal):
    if start == goal:
        return [start]

    # Inicializar frentes de búsqueda y conjuntos visitados
    frontera_adelante = deque([(start, [start])])
    frontera_atras = deque([(goal, [goal])])
    visitados_adelante = {start: [start]}
    visitados_atras = {goal: [goal]}

    while frontera_adelante and frontera_atras:
        # Expande la búsqueda desde el inicio
        camino_adelante = expandir_frontera(graph, frontera_adelante, visitados_adelante, visitados_atras)
        if camino_adelante:
            return camino_adelante

        # Expande la búsqueda desde el objetivo
        camino_atras = expandir_frontera(graph, frontera_atras, visitados_atras, visitados_adelante)
        if camino_atras:
            return camino_atras[::-1]

    return None

def expandir_frontera(graph, cola, visitados, otros_visitados):
    for _ in range(len(cola)):
        nodo_actual, ruta_actual = cola.popleft()

        for vecino in graph[nodo_actual]:
            if vecino not in visitados:
                nueva_ruta = ruta_actual + [vecino]
                visitados[vecino] = nueva_ruta
                cola.append((vecino, nueva_ruta))

                # Verifica si hay intersección con la otra búsqueda
                if vecino in otros_visitados:
                    return ruta_actual + otros_visitados[vecino][::-1]
    return None

# Ejemplo de grafo
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Ejecutar la búsqueda bidireccional
path = bidirectional_search(graph, 'A', 'F')
print(f"Camino encontrado con búsqueda bidireccional: {path}")
