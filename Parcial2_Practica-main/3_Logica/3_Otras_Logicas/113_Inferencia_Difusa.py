import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Importamos las bibliotecas necesarias

# Definimos las variables difusas
# En este caso, modelaremos un sistema para recomendar libros basado en el interés del lector y la dificultad del libro

# Entrada difusa: Interés del lector (0 a 10)
interes = ctrl.Antecedent(np.arange(0, 11, 1), 'interes')
# Entrada difusa: Dificultad del libro (0 a 10)
dificultad = ctrl.Antecedent(np.arange(0, 11, 1), 'dificultad')
# Salida difusa: Recomendación (0 a 10)
recomendacion = ctrl.Consequent(np.arange(0, 11, 1), 'recomendacion')

# Definimos las funciones de pertenencia para cada variable
# Interés del lector
interes['bajo'] = fuzz.trimf(interes.universe, [0, 0, 5])
interes['medio'] = fuzz.trimf(interes.universe, [0, 5, 10])
interes['alto'] = fuzz.trimf(interes.universe, [5, 10, 10])

# Dificultad del libro
dificultad['facil'] = fuzz.trimf(dificultad.universe, [0, 0, 5])
dificultad['moderada'] = fuzz.trimf(dificultad.universe, [0, 5, 10])
dificultad['dificil'] = fuzz.trimf(dificultad.universe, [5, 10, 10])

# Recomendación
recomendacion['baja'] = fuzz.trimf(recomendacion.universe, [0, 0, 5])
recomendacion['media'] = fuzz.trimf(recomendacion.universe, [0, 5, 10])
recomendacion['alta'] = fuzz.trimf(recomendacion.universe, [5, 10, 10])

# Definimos las reglas difusas
rule1 = ctrl.Rule(interes['bajo'] & dificultad['dificil'], recomendacion['baja'])
rule2 = ctrl.Rule(interes['medio'] & dificultad['moderada'], recomendacion['media'])
rule3 = ctrl.Rule(interes['alto'] & dificultad['facil'], recomendacion['alta'])
rule4 = ctrl.Rule(interes['alto'] & dificultad['dificil'], recomendacion['media'])
rule5 = ctrl.Rule(interes['bajo'] & dificultad['facil'], recomendacion['media'])

# Creamos el sistema de control difuso
recomendacion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
recomendacion_simulador = ctrl.ControlSystemSimulation(recomendacion_ctrl)

# Simulamos el sistema con valores de entrada
# Por ejemplo, interés = 7 y dificultad = 4
recomendacion_simulador.input['interes'] = 7
recomendacion_simulador.input['dificultad'] = 4

# Calculamos la salida
recomendacion_simulador.compute()

# Mostramos el resultado
print(f"Nivel de recomendación: {recomendacion_simulador.output['recomendacion']:.2f}")

# Visualizamos las gráficas de las funciones de pertenencia
interes.view()
dificultad.view()
recomendacion.view()
