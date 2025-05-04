from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana para la toma de decisiones del agente
modelo_decision = BayesianNetwork([
    ('EnemigoCerca', 'Decision'),
    ('Salud', 'Decision'),
    ('Visibilidad', 'Decision')
])

# Definir las tablas de probabilidad condicional (CPD) en español
cpd_enemigo_cerca = TabularCPD(variable='EnemigoCerca', variable_card=2, values=[[0.7], [0.3]])
cpd_salud = TabularCPD(variable='Salud', variable_card=3, values=[[0.4], [0.4], [0.2]])
cpd_visibilidad = TabularCPD(variable='Visibilidad', variable_card=2, values=[[0.6], [0.4]])

# Probabilidades condicionales para la decisión del agente
cpd_decision_agente = TabularCPD(variable='Decision', variable_card=3,
                                 values=[
                                     # EnemigoCerca, Salud, Visibilidad
                                     [0.8, 0.1, 0.1, 0.7, 0.2, 0.1, 0.6, 0.3, 0.1, 0.5, 0.4, 0.1],  # Atacar
                                     [0.1, 0.7, 0.2, 0.2, 0.6, 0.2, 0.2, 0.5, 0.3, 0.3, 0.5, 0.2],  # Defender
                                     [0.1, 0.2, 0.7, 0.1, 0.2, 0.7, 0.2, 0.2, 0.6, 0.2, 0.1, 0.7]    # Explorar
                                 ],
                                 evidence=['EnemigoCerca', 'Salud', 'Visibilidad'],
                                 evidence_card=[2, 3, 2])

# Asociar las CPDs al modelo
modelo_decision.add_cpds(cpd_enemigo_cerca, cpd_salud, cpd_visibilidad, cpd_decision_agente)

# Verificar si el modelo es válido
assert modelo_decision.check_model()

# Realizar inferencia utilizando eliminación de variables
inferencia = VariableElimination(modelo_decision)

# Probabilidades de cada decisión dado el estado del entorno y del agente
evidencia = {'EnemigoCerca': 1, 'Salud': 2, 'Visibilidad': 1}  # Ejemplo: enemigo cerca, salud media, visibilidad alta
probabilidades_decision = inferencia.query(variables=['Decision'], evidence=evidencia)
print("Probabilidades de cada decisión del agente:")
print(probabilidades_decision)

# Tomar la decisión con mayor probabilidad
mejor_decision_indice = probabilidades_decision.values.argmax()
mapa_decision = {0: 'Atacar', 1: 'Defender', 2: 'Explorar'}
print(f"Mejor decisión para el agente: {mapa_decision[mejor_decision_indice]}")
