def es_amenaza(tablero, fila, columna, tamano):
    # Comprobar la columna hacia arriba
    for i in range(fila):
        if tablero[i][columna] == 1:
            return True

    # Comprobar la diagonal superior izquierda
    for i, j in zip(range(fila, -1, -1), range(columna, -1, -1)):
        if tablero[i][j] == 1:
            return True

    # Comprobar la diagonal superior derecha
    for i, j in zip(range(fila, -1, -1), range(columna, tamano)):
        if tablero[i][j] == 1:
            return True

    return False

def resolver_n_reinas_backtracking_inteligente(tablero, fila_actual, tamano, conflictos_previos):
    if fila_actual >= tamano:
        return True

    for columna in range(tamano):
        if not es_amenaza(tablero, fila_actual, columna, tamano):
            tablero[fila_actual][columna] = 1
            if resolver_n_reinas_backtracking_inteligente(tablero, fila_actual + 1, tamano, conflictos_previos):
                return True
            tablero[fila_actual][columna] = 0
            conflictos_previos[fila_actual] = columna  # Registrar posible conflicto

    # Backjumping implícito al no encontrar columna segura
    return False

def encontrar_solucion_n_reinas_backjumping(num_reinas):
    tablero = [[0 for _ in range(num_reinas)] for _ in range(num_reinas)]
    historial_conflictos = [-1] * num_reinas
    if not resolver_n_reinas_backtracking_inteligente(tablero, 0, num_reinas, historial_conflictos):
        print("No se encontró una disposición válida.")
        return None
    return tablero

# Ejemplo de uso
num_reinas = 4
solucion = encontrar_solucion_n_reinas_backjumping(num_reinas)

if solucion:
    print(f"Disposición de {num_reinas} reinas con backjumping:")
    for fila in solucion:
        print(fila)
