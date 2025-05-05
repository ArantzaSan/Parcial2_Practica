import random

def cadena_de_markov_montecarlo(estados_clima, matriz_transicion, estado_inicial, pasos):
    """
    Simula las probabilidades del clima utilizando la Cadena de Markov Montecarlo.

    :param estados_clima: Lista de posibles estados del clima (ej., ["Soleado", "Lluvioso", "Nublado"])
    :param matriz_transicion: Matriz de probabilidades de transición entre estados
    :param estado_inicial: Estado del clima inicial
    :param pasos: Número de pasos a simular
    :return: Diccionario con las probabilidades de cada estado del clima
    """
    estado_actual = estado_inicial
    conteo_estados = {estado: 0 for estado in estados_clima}

    for _ in range(pasos):
        conteo_estados[estado_actual] += 1
        siguiente_estado = random.choices(
            estados_clima, weights=matriz_transicion[estados_clima.index(estado_actual)]
        )[0]
        estado_actual = siguiente_estado

    total_pasos = sum(conteo_estados.values())
    probabilidades = {estado: conteo / total_pasos for estado, conteo in conteo_estados.items()}
    return probabilidades


if __name__ == "__main__":
    # Definir los estados del clima y las probabilidades de transición
    estados_clima = ["Soleado", "Lluvioso", "Nublado"]
    matriz_transicion = [
        [0.7, 0.2, 0.1],  # Probabilidades desde "Soleado"
        [0.3, 0.4, 0.3],  # Probabilidades desde "Lluvioso"
        [0.2, 0.3, 0.5],  # Probabilidades desde "Nublado"
    ]

    # Estado inicial y número de pasos
    estado_inicial = "Soleado"
    pasos = 10000

    # Ejecutar la simulación
    probabilidades_clima = cadena_de_markov_montecarlo(estados_clima, matriz_transicion, estado_inicial, pasos)

    # Imprimir los resultados
    print("Probabilidades del clima después de la simulación:")
    for estado, prob in probabilidades_clima.items():
        print(f"{estado}: {prob:.4f}")
