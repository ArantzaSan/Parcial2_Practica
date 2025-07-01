import numpy as np

# 123_Modelo_Proba_Racional.py
# Implementación de un modelo de probabilidad racional
# Autor: GitHub Copilot


# ------------------------------
# 1. Definición del problema
# Supongamos que queremos modelar la probabilidad racional de elegir una opción
# entre varias alternativas, dado un conjunto de evidencias observadas.
# Usaremos el modelo de elección racional basado en la probabilidad de máxima utilidad.

# ------------------------------
# 2. Definición de utilidades y alternativas

# Lista de alternativas posibles
alternativas = ['A', 'B', 'C']

# Utilidades asociadas a cada alternativa (pueden venir de datos o estimaciones)
utilidades = np.array([3.0, 5.0, 2.0])

# ------------------------------
# 3. Cálculo de probabilidades racionales (softmax)

def probabilidad_racional(utilidades, temperatura=1.0):
    """
    Calcula la probabilidad racional de elegir cada alternativa usando softmax.
    temperatura: controla el grado de aleatoriedad (más baja = más determinista)
    """
    utilidades_ajustadas = utilidades / temperatura
    exp_utilidades = np.exp(utilidades_ajustadas - np.max(utilidades_ajustadas))  # para estabilidad numérica
    probabilidades = exp_utilidades / np.sum(exp_utilidades)
    return probabilidades

# ------------------------------
# 4. Ejecución del modelo

# Parámetro de temperatura (ajustable)
temperatura = 1.0

# Calculamos las probabilidades racionales
probs = probabilidad_racional(utilidades, temperatura)

# ------------------------------
# 5. Presentación de resultados

print("Alternativas y probabilidades racionales de elección:")
for alt, p in zip(alternativas, probs):
    print(f"  {alt}: {p:.3f}")

# ------------------------------
# 6. Ejemplo de simulación de elección

def elegir_alternativa(alternativas, probabilidades):
    """
    Simula la elección de una alternativa según las probabilidades dadas.
    """
    return np.random.choice(alternativas, p=probabilidades)

# Simulamos una elección
eleccion = elegir_alternativa(alternativas, probs)
print(f"\nAlternativa elegida (simulación): {eleccion}")
