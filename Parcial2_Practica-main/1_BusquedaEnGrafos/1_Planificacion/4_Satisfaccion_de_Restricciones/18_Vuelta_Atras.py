def es_posicion_valida(tablero, fila, columna, tamano):
    # Comprobar la columna actual hacia arriba
    for i in range(fila):
        if tablero[i][columna] == 'Q':
            return False

    # Comprobar la diagonal superior izquierda
    for i, j in zip(range(fila, -1, -1), range(columna, -1, -1)):
        if tablero[i][j] == 'Q':
            return False

    # Comprobar la diagonal superior derecha
    for i, j in zip(range(fila, -1, -1), range(columna, tamano)):
        if tablero[i][j] == 'Q':
            return False

    return True

def resolver_n_reinas_recursivo(tablero, fila_actual, tamano):
    if fila_actual >= tamano:
        return True

    for columna in range(tamano):
        if es_posicion_valida(tablero, fila_actual, columna, tamano):
            tablero[fila_actual][columna] = 'Q'
            if resolver_n_reinas_recursivo(tablero, fila_actual + 1, tamano):
                return True
            tablero[fila_actual][columna] = '.'  # Backtracking

    return False

def encontrar_solucion_n_reinas(num_reinas):
    tablero = [['.' for _ in range(num_reinas)] for _ in range(num_reinas)]
    if not resolver_n_reinas_recursivo(tablero, 0, num_reinas):
        print("No se encontró una disposición válida.")
        return None
    return tablero

# Ejemplo de uso
num_reinas = 4
solucion = encontrar_solucion_n_reinas(num_reinas)

if solucion:
    print(f"Disposición de {num_reinas} reinas:")
    for fila in solucion:
        print(' '.join(fila))
