import random

def evaluar_solucion(solucion):
    # Función de evaluación: maximizar el producto de los elementos
    producto = 1
    for x in solucion:
        producto *= x
    return producto

def generar_variantes(solucion_actual):
    # Genera variantes incrementando o decrementando aleatoriamente un elemento
    variantes = []
    for i in range(len(solucion_actual)):
        variante_incremento = list(solucion_actual)
        variante_incremento[i] += 1
        variantes.append(variante_incremento)

        variante_decremento = list(solucion_actual)
        variante_decremento[i] -= 1
        variantes.append(variante_decremento)
    return variantes

def busqueda_de_haz_localizada(estado_inicial, ancho_haz=4, max_iter=120):
    # Inicializar el haz de soluciones
    haz_actual = [list(estado_inicial)]
    mejor_solucion_global = list(estado_inicial)
    mejor_valor_global = evaluar_solucion(estado_inicial)

    for _ in range(max_iter):
        candidatos = []
        for estado in haz_actual:
            vecinos = generar_variantes(estado)
            candidatos.extend(vecinos)

        # Evaluar y seleccionar los mejores estados para el nuevo haz
        candidatos.sort(key=evaluar_solucion, reverse=True)  # Maximizar el producto
        haz_actual = candidatos[:ancho_haz]

        # Actualizar la mejor solución global encontrada
        for estado in haz_actual:
            valor_actual = evaluar_solucion(estado)
            if valor_actual > mejor_valor_global:
                mejor_solucion_global = list(estado)
                mejor_valor_global = valor_actual

        if not haz_actual:  # Si el haz se vacía
            break

    return mejor_solucion_global, mejor_valor_global

# Estado inicial
estado_inicial = [1, 2, 3, 4]

# Ejecutar la búsqueda de haz local
solucion_final, valor_final = busqueda_de_haz_localizada(estado_inicial)
print(f"Solución final encontrada: {solucion_final} con valor de evaluación: {valor_final}")
