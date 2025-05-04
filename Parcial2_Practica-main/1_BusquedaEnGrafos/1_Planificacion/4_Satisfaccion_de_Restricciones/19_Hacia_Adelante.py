def es_segura_reina(tablero, fila, columna, tamano):
    # Verificar columna
    for i in range(fila):
        if tablero[i][columna] == 1:
            return False

    # Verificar diagonal superior izquierda
    for i, j in zip(range(fila, -1, -1), range(columna, -1, -1)):
        if tablero[i][j] == 1:
            return False

    # Verificar diagonal superior derecha
    for i, j in zip(range(fila, -1, -1), range(columna, tamano)):
        if tablero[i][j] == 1:
            return False

    return True

def propagacion_restricciones(tablero, fila_actual, tamano, dominios):
    if fila_actual >= tamano:
        return True

    for columna in list(dominios[fila_actual]):
        if es_segura_reina(tablero, fila_actual, columna, tamano):
            tablero[fila_actual][columna] = 1
            nuevos_dominios = [set(d) for d in dominios]

            # Propagar restricciones a las filas siguientes
            for i in range(fila_actual + 1, tamano):
                if columna in nuevos_dominios[i]:
                    nuevos_dominios[i].discard(columna)
                diag_izq = columna - (i - fila_actual)
                if 0 <= diag_izq < tamano and diag_izq in nuevos_dominios[i]:
                    nuevos_dominios[i].discard(diag_izq)
                diag_der = columna + (i - fila_actual)
                if 0 <= diag_der < tamano and diag_der in nuevos_dominios[i]:
                    nuevos_dominios[i].discard(diag_der)

            if all(nuevos_dominios[i] for i in range(fila_actual + 1, tamano)):
                if propagacion_restricciones(tablero, fila_actual + 1, tamano, nuevos_dominios):
                    return True
            tablero[fila_actual][columna] = 0

    return False

def n_reinas_con_propagacion(n):
    tablero = [[0 for _ in range(n)] for _ in range(n)]
    dominios = [set(range(n)) for _ in range(n)]
    if not propagacion_restricciones(tablero, 0, n, dominios):
        print("No existe solución.")
        return None
    return tablero

# Ejemplo de uso
num_reinas = 4
solucion = n_reinas_con_propagacion(num_reinas)

if solucion:
    print(f"Solución para {num_reinas}-reinas con propagación de restricciones:")
    for fila in solucion:
        print(fila)
