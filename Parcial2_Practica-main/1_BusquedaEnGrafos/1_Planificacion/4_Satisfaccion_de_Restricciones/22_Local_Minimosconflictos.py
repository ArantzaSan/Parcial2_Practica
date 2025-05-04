import random

def contar_amenazas(tablero, fila, columna, tamano):
    amenazas = 0
    # Revisar columna
    for i in range(tamano):
        if i != fila and tablero[i][columna] == 1:
            amenazas += 1

    # Revisar diagonal superior izquierda
    i, j = fila - 1, columna - 1
    while i >= 0 and j >= 0:
        if tablero[i][j] == 1:
            amenazas += 1
        i -= 1
        j -= 1

    # Revisar diagonal superior derecha
    i, j = fila - 1, columna + 1
    while i >= 0 and j < tamano:
        if tablero[i][j] == 1:
            amenazas += 1
        i -= 1
        j += 1

    # Revisar diagonal inferior izquierda
    i, j = fila + 1, columna - 1
    while i < tamano and j >= 0:
        if tablero[i][j] == 1:
            amenazas += 1
        i += 1
        j -= 1

    # Revisar diagonal inferior derecha
    i, j = fila + 1, columna + 1
    while i < tamano and j < tamano:
        if tablero[i][j] == 1:
            amenazas += 1
        i += 1
        j += 1

    return amenazas

def resolver_minimos_conflictos(tablero, tamano, max_iteraciones=1000):
    # Inicializar el tablero con una reina por fila en columnas aleatorias
    for i in range(tamano):
        tablero[i][random.randint(0, tamano - 1)] = 1

    for _ in range(max_iteraciones):
        # Identificar reinas en conflicto
        reinas_en_conflicto = [(r, c) for r in range(tamano) for c in range(tamano) if tablero[r][c] == 1 and contar_amenazas(tablero, r, c, tamano) > 0]

        if not reinas_en_conflicto:
            return tablero

        fila, columna_actual = random.choice(reinas_en_conflicto)
        tablero[fila][columna_actual] = 0

        # Encontrar la columna con la menor cantidad de amenazas para esta fila
        mejor_columna = min(range(tamano), key=lambda c: contar_amenazas(tablero, fila, c, tamano))
        tablero[fila][mejor_columna] = 1

    return tablero

def n_reinas_min_conflictos_local(n):
    tablero = [[0 for _ in range(n)] for _ in range(n)]
    solucion = resolver_min
