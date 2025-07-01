# 138_Mejor_Hipotesis_Actual.py
# Implementación del algoritmo de la Mejor Hipótesis Actual (Versión Filosófica)
# Este programa simula cómo, en filosofía de la ciencia, se ajustan las hipótesis a la evidencia disponible.

# 1. Definición de la estructura de una hipótesis
# En este ejemplo, una hipótesis es una lista de características (atributos) que pueden ser específicas o generales.

def inicializar_hipotesis(num_atributos):
    # Inicialmente, la hipótesis es la más general posible (todo es '?')
    return ['?' for _ in range(num_atributos)]

# 2. Función para actualizar la hipótesis con base en un ejemplo positivo
# Si la hipótesis es demasiado general, se especializa para ajustarse al ejemplo.

def actualizar_hipotesis(hipotesis, ejemplo):
    for i in range(len(hipotesis)):
        if hipotesis[i] == '?':
            hipotesis[i] = ejemplo[i]
        elif hipotesis[i] != ejemplo[i]:
            hipotesis[i] = '?'
    return hipotesis

# 3. Simulación de observaciones (ejemplos positivos)
# Cada ejemplo representa un caso observado en la "realidad" filosófica.

ejemplos = [
    ['rojo', 'redondo', 'grande'],
    ['rojo', 'ovalado', 'grande'],
    ['rojo', 'redondo', 'grande']
]

# 4. Proceso de aprendizaje: encontrar la mejor hipótesis actual
# Se parte de la hipótesis más general y se va ajustando con cada ejemplo.

num_atributos = len(ejemplos[0])
hipotesis_actual = inicializar_hipotesis(num_atributos)

for ejemplo in ejemplos:
    hipotesis_actual = actualizar_hipotesis(hipotesis_actual, ejemplo)

# 5. Presentación del resultado
# La hipótesis final es la mejor explicación actual de los datos observados.

print("Mejor hipótesis actual (filosóficamente hablando):")
print(hipotesis_actual)

# 6. Reflexión filosófica (comentario)
# Así como en la filosofía de la ciencia, nuestras hipótesis se ajustan a la evidencia,
# pero siempre pueden cambiar si aparecen nuevos ejemplos que desafíen lo que creemos saber.
