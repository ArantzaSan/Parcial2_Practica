import heapq
from collections import namedtuple

# Representación explícita de las aristas
Arista = namedtuple('Arista', ['destino', 'peso', 'tipo'])

def calcular_heuristica(estado, objetivo):
    # Heurística cuadrática de la distancia euclidiana
    dx = estado[0] - objetivo[0]
    dy = estado[1] - objetivo[1]
    return dx*dx + dy*dy

def busqueda_ao_estrella_modificada(grafo, inicio, fin):
    # Prioridad de exploración: (f_score, estado)
    cola_prioritaria = [(calcular_heuristica(inicio, fin), inicio)]
    # Costo real acumulado desde el inicio
    costo_real_hasta = {inicio: 0}
    # Tipo de nodo inferido durante la búsqueda
    tipo_de_nodo = {inicio: 'OR'}
    # Nodos que ya han sido evaluados completamente
    nodos_cerrados = set()

    while cola_prioritaria:
        # Extraer el nodo con la menor f_score
        f_score, nodo_actual = heapq.heappop(cola_prioritaria)

        # Si el nodo actual es el objetivo, retornar el costo
        if nodo_actual == fin:
            return costo_real_hasta[nodo_actual]

        # Marcar el nodo como cerrado
        nodos_cerrados.add(nodo_actual)

        # Examinar las aristas salientes
        for arista in grafo.get(nodo_actual, []):
            vecino = arista.destino
            peso = arista.peso
            tipo = arista.tipo

            nuevo_costo = costo_real_hasta[nodo_actual] + peso

            if vecino not in costo_real_hasta or nuevo_costo < costo_real_hasta[vecino]:
                costo_real_hasta[vecino] = nuevo_costo
                prioridad = nuevo_costo + calcular_heuristica(vecino, fin)
                heapq.heappush(cola_prioritaria, (prioridad, vecino))
                tipo_de_nodo[vecino] = tipo

            # Manejo específico para nodos AND (se evalúan todos los sucesores)
            if tipo_de_nodo[nodo_actual] == 'AND':
                costo_and_temporal = 0
                todos_visitados = True
                for arista_and in grafo.get(nodo_actual, []):
                    if arista_and.destino not in costo_real_hasta:
                        todos_visitados = False
                        break
                    costo_and_temporal += costo_real_hasta[arista_and.destino] + arista_and.peso
                if todos_visitados and nodo_actual in costo_real_hasta:
                    costo_real_hasta[nodo_actual] = costo_and_temporal


    return None

# Definición del grafo AND/OR usando la estructura Arista
grafo_ao_alternativo = {
    (0, 0): [Arista((1, 0), 1, 'OR'), Arista((0, 1), 1, 'AND')],
    (1, 0): [Arista((1, 1), 1, 'OR')],
    (0, 1): [Arista((1, 1), 1, 'OR')],
    (1, 1): [Arista((1, 2), 1, 'OR')],
    (1, 2): []
}

# Ejecutar la búsqueda AO* modificada
costo_final_alternativo = busqueda_ao_estrella_modificada(grafo_ao_alternativo, (0, 0), (1, 2))
print(f"Costo de la solución desde (0, 0) hasta (1, 2) usando AO* (alternativo): {costo_final_alternativo}")
