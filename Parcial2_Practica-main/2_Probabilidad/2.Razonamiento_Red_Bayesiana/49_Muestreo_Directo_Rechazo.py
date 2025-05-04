import random

def muestreo_directo(probabilidades):
    """
    Realiza un muestreo directo basado en una distribución de probabilidad.
    :param probabilidades: Lista de probabilidades asociadas a cada evento.
    :return: Índice del evento seleccionado.
    """
    acumulada = [sum(probabilidades[:i+1]) for i in range(len(probabilidades))]
    r = random.uniform(0, 1)
    for i, prob in enumerate(acumulada):
        if r <= prob:
            return i

# Ejemplo: Probabilidades de mezclas químicas
mezclas = ["Mezcla A", "Mezcla B", "Mezcla C"]
probabilidades = [0.5, 0.3, 0.2]  # Probabilidades asociadas a cada mezcla

# Validación de que las probabilidades sumen 1
if not abs(sum(probabilidades) - 1.0) < 1e-6:
    raise ValueError("Las probabilidades deben sumar 1.")

# Realizar muestreo directo
resultados = {mezcla: 0 for mezcla in mezclas}
n_muestras = 1000

for _ in range(n_muestras):
    indice = muestreo_directo(probabilidades)
    resultados[mezclas[indice]] += 1

# Mostrar resultados
print("Resultados del muestreo directo:")
for mezcla, conteo in resultados.items():
    print(f"{mezcla}: {conteo} muestras ({(conteo / n_muestras) * 100:.2f}%)")