import random

def muestreo_simple(probs):
    """
    Realiza un muestreo directo basado en una distribución de probabilidad discreta.
    :param probs: Lista de probabilidades correspondientes a cada resultado posible.
    :return: Índice del resultado seleccionado.
    """
    prob_acumulada = 0.0
    rand_num = random.random()
    for indice, probabilidad in enumerate(probs):
        prob_acumulada += probabilidad
        if rand_num <= prob_acumulada:
            return indice
    return len(probs) - 1 # Por si acaso hay errores de redondeo

# Ejemplo: Selección de componentes químicos
componentes = ["Componente X", "Componente Y", "Componente Z"]
probabilidades_comp = [0.4, 0.35, 0.25]  # Probabilidades de selección de cada componente

# Verificar que las probabilidades sean válidas
if not abs(sum(probabilidades_comp) - 1.0) < 1e-9:
    raise ValueError("La suma de las probabilidades debe ser igual a 1.")

# Ejecutar el muestreo simple
resultados_muestreo = {comp: 0 for comp in componentes}
numero_muestras = 1500

for _ in range(numero_muestras):
    seleccion = muestreo_simple(probabilidades_comp)
    resultados_muestreo[componentes[seleccion]] += 1

# Imprimir los resultados
print("Resultados del muestreo simple:")
for componente, frecuencia in resultados_muestreo.items():
    porcentaje = (frecuencia / numero_muestras) * 100
    print(f"{componente}: {frecuencia} ocurrencias ({porcentaje:.2f}%)")
