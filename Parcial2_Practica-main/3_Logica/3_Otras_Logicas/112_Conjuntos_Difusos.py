import skfuzzy as fuzz
import numpy as np

# Importamos la librería necesaria para trabajar con lógica difusa

# Definimos las variables difusas y sus universos de discurso
# En este caso, trabajaremos con un ejemplo matemático: calificación de estudiantes
# Universo de discurso: 0 a 100 (calificaciones)
x_calificaciones = np.arange(0, 101, 1)

# Definimos las funciones de pertenencia para las categorías de calificaciones
# Categorías: "Baja", "Media", "Alta"
calificacion_baja = fuzz.trapmf(x_calificaciones, [0, 0, 30, 50])
calificacion_media = fuzz.trimf(x_calificaciones, [40, 60, 80])
calificacion_alta = fuzz.trapmf(x_calificaciones, [70, 90, 100, 100])

# Mostramos las funciones de pertenencia
import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_calificaciones, calificacion_baja, label='Baja')
plt.plot(x_calificaciones, calificacion_media, label='Media')
plt.plot(x_calificaciones, calificacion_alta, label='Alta')
plt.title('Funciones de Pertenencia de Calificaciones')
plt.xlabel('Calificación')
plt.ylabel('Grado de Pertenencia')
plt.legend()
plt.grid()
plt.show()

# Definimos un ejemplo práctico: calcular la pertenencia de una calificación específica
# Calificación de ejemplo
calificacion_ejemplo = 75

# Calculamos el grado de pertenencia a cada categoría
grado_baja = fuzz.interp_membership(x_calificaciones, calificacion_baja, calificacion_ejemplo)
grado_media = fuzz.interp_membership(x_calificaciones, calificacion_media, calificacion_ejemplo)
grado_alta = fuzz.interp_membership(x_calificaciones, calificacion_alta, calificacion_ejemplo)

# Mostramos los resultados
print(f"Calificación: {calificacion_ejemplo}")
print(f"Grado de pertenencia a 'Baja': {grado_baja:.2f}")
print(f"Grado de pertenencia a 'Media': {grado_media:.2f}")
print(f"Grado de pertenencia a 'Alta': {grado_alta:.2f}")

# Ejemplo de inferencia difusa: determinar si un estudiante aprueba o no
# Reglas:
# - Si la calificación es "Alta", entonces "Aprueba"
# - Si la calificación es "Media", entonces "Aprueba Parcialmente"
# - Si la calificación es "Baja", entonces "No Aprueba"

# Grados de pertenencia a las salidas
aprueba = grado_alta
aprueba_parcialmente = grado_media
no_aprueba = grado_baja

# Mostramos los resultados de la inferencia
print("\nResultados de la Inferencia Difusa:")
print(f"Grado de pertenencia a 'Aprueba': {aprueba:.2f}")
print(f"Grado de pertenencia a 'Aprueba Parcialmente': {aprueba_parcialmente:.2f}")
print(f"Grado de pertenencia a 'No Aprueba': {no_aprueba:.2f}")
