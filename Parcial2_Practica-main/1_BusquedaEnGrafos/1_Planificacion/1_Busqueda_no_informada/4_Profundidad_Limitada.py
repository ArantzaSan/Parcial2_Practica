def explorar_profundamente_con_tope(grafo, inicio, meta, max_profundidad):
    """
    Busca un camino desde un nodo inicial a un nodo objetivo en un grafo,
    limitando la profundidad de la búsqueda.

    Args:
        grafo (dict): Representación del grafo (adyacencia).
        inicio: Nodo de inicio de la búsqueda.
        meta: Nodo objetivo a encontrar.
        max_profundidad (int): Límite máximo de profundidad a explorar.

    Returns:
        list or None: Camino encontrado (lista de nodos) o None si no se encuentra.
    """
    ruta_actual = [inicio]
    pila = [(inicio, ruta_actual, 0)]  # (nodo, camino actual, profundidad)
    visitados_en_rama = {inicio}

    while pila:
        nodo, camino, profundidad = pila.pop()

        if nodo == meta:
            return camino

        if profundidad < max_profundidad:
            sucesores = grafo.get(nodo, [])
            for vecino in sucesores:
                if vecino not in visitados_en_rama:
                    visitados_en_rama.add(vecino)
                    nueva_ruta = list(camino)  # Crear una nueva lista para evitar modificaciones
                    nueva_ruta.append(vecino)
                    pila.append((vecino, nueva_ruta, profundidad + 1))
            # Al retroceder en la búsqueda, removemos el nodo del conjunto de visitados en la rama
            if pila:
                ultimo_en_pila, _, _ = pila[-1]
                visitados_en_rama = set(camino[:-1]) if camino[:-1] else set()
                if ultimo_en_pila in grafo.get(camino[-2], []) if len(camino) > 1 else False:
                    visitados_en_rama.add(camino[-2])


    return None

# Definición del grafo
estructura_datos_grafo = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# Buscar camino con límite de profundidad
resultado = explorar_profundamente_con_tope(estructura_datos_grafo, 'A', 'F', 3)
print(f"Resultado de la búsqueda con profundidad máxima: {resultado}")
