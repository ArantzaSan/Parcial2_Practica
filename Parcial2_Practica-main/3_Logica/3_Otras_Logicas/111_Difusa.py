import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Importamos la librería necesaria para trabajar con lógica difusa

# Definimos las variables difusas
# Entrada: Popularidad de la película (0 a 10)
popularidad = ctrl.Antecedent(range(0, 11), 'popularidad')
# Entrada: Presupuesto del cliente (0 a 100)
presupuesto = ctrl.Antecedent(range(0, 101), 'presupuesto')
# Salida: Probabilidad de ir al cine (0 a 100)
probabilidad_ir = ctrl.Consequent(range(0, 101), 'probabilidad_ir')

# Definimos las funciones de pertenencia para cada variable
# Popularidad: baja, media, alta
popularidad['baja'] = fuzz.trimf(popularidad.universe, [0, 0, 5])
popularidad['media'] = fuzz.trimf(popularidad.universe, [0, 5, 10])
popularidad['alta'] = fuzz.trimf(popularidad.universe, [5, 10, 10])

# Presupuesto: bajo, medio, alto
presupuesto['bajo'] = fuzz.trimf(presupuesto.universe, [0, 0, 50])
presupuesto['medio'] = fuzz.trimf(presupuesto.universe, [25, 50, 75])
presupuesto['alto'] = fuzz.trimf(presupuesto.universe, [50, 100, 100])

# Probabilidad de ir al cine: baja, media, alta
probabilidad_ir['baja'] = fuzz.trimf(probabilidad_ir.universe, [0, 0, 50])
probabilidad_ir['media'] = fuzz.trimf(probabilidad_ir.universe, [25, 50, 75])
probabilidad_ir['alta'] = fuzz.trimf(probabilidad_ir.universe, [50, 100, 100])

# Definimos las reglas difusas
rule1 = ctrl.Rule(popularidad['baja'] & presupuesto['bajo'], probabilidad_ir['baja'])
rule2 = ctrl.Rule(popularidad['media'] & presupuesto['medio'], probabilidad_ir['media'])
rule3 = ctrl.Rule(popularidad['alta'] & presupuesto['alto'], probabilidad_ir['alta'])
rule4 = ctrl.Rule(popularidad['alta'] & presupuesto['medio'], probabilidad_ir['media'])
rule5 = ctrl.Rule(popularidad['media'] & presupuesto['alto'], probabilidad_ir['alta'])

# Creamos el sistema de control difuso
cine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
cine_simulador = ctrl.ControlSystemSimulation(cine_ctrl)

# Simulamos el sistema con valores de entrada
cine_simulador.input['popularidad'] = 8  # Popularidad de la película
cine_simulador.input['presupuesto'] = 60  # Presupuesto del cliente

# Calculamos la salida
cine_simulador.compute()

# Mostramos el resultado
print(f"Probabilidad de ir al cine: {cine_simulador.output['probabilidad_ir']:.2f}%")
