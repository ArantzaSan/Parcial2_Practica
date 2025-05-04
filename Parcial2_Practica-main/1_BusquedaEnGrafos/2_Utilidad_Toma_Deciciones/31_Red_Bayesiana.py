from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana
model = BayesianNetwork([
    ('EnemyNear', 'Decision'),
    ('Health', 'Decision'),
    ('Visibility', 'Decision')
])

# Definir las tablas de probabilidad condicional (CPD)
cpd_enemy_near = TabularCPD(variable='EnemyNear', variable_card=2, values=[[0.7], [0.3]])
cpd_health = TabularCPD(variable='Health', variable_card=3, values=[[0.4], [0.4], [0.2]])
cpd_visibility = TabularCPD(variable='Visibility', variable_card=2, values=[[0.6], [0.4]])

# Probabilidades condicionales para la decisión
cpd_decision = TabularCPD(variable='Decision', variable_card=3,
                          values=[
                              # EnemyNear, Health, Visibility
                              [0.8, 0.1, 0.1, 0.7, 0.2, 0.1, 0.6, 0.3, 0.1, 0.5, 0.4, 0.1],  # Attack
                              [0.1, 0.7, 0.2, 0.2, 0.6, 0.2, 0.2, 0.5, 0.3, 0.3, 0.5, 0.2],  # Defend
                              [0.1, 0.2, 0.7, 0.1, 0.2, 0.7, 0.2, 0.2, 0.6, 0.2, 0.1, 0.7]   # Explore
                          ],
                          evidence=['EnemyNear', 'Health', 'Visibility'],
                          evidence_card=[2, 3, 2])

# Asociar las CPDs al modelo
model.add_cpds(cpd_enemy_near, cpd_health, cpd_visibility, cpd_decision)

# Verificar si el modelo es válido
assert model.check_model()

# Realizar inferencia utilizando eliminación de variables
infer = VariableElimination(model)

# Probabilidades de cada decisión dado el estado del entorno y del personaje
evidence = {'EnemyNear': 1, 'Health': 2, 'Visibility': 1}  # Ejemplo: enemigo cerca, salud media, visibilidad alta
decision_prob = infer.query(variables=['Decision'], evidence=evidence)
print("Probabilidades de cada decisión:")
print(decision_prob)

# Tomar la decisión con mayor probabilidad
best_decision = decision_prob.values.argmax()
decision_map = {0: 'Attack', 1: 'Defend', 2: 'Explore'}
print(f"Mejor decisión: {decision_map[best_decision]}")
