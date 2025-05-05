from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Estructura de la red bayesiana
modelo = BayesianNetwork([('Lluvia', 'Charcos'),
                          ('Aspersor', 'Charcos'),
                          ('Lluvia', 'Cesped_Mojado'),
                          ('Charcos', 'Cesped_Mojado')])

# Distribuciones de probabilidad condicional
dp_lluvia = TabularCPD(variable='Lluvia', variable_card=2, values=[[0.7], [0.3]])
dp_aspersor = TabularCPD(variable='Aspersor', variable_card=2, values=[[0.6], [0.4]])
dp_charcos = TabularCPD(variable='Charcos', variable_card=2,
                         values=[[0.9, 0.4, 0.3, 0.1],
                                 [0.1, 0.6, 0.7, 0.9]],
                         evidence=['Lluvia', 'Aspersor'], evidence_card=[2, 2])
dp_cesped_mojado = TabularCPD(variable='Cesped_Mojado', variable_card=2,
                               values=[[0.8, 0.2, 0.1, 0.05],
                                       [0.2, 0.8, 0.9, 0.95]],
                               evidence=['Lluvia', 'Charcos'], evidence_card=[2, 2])

# Añadir las distribuciones al modelo
modelo.add_cpds(dp_lluvia, dp_aspersor, dp_charcos, dp_cesped_mojado)

# Verificar la validez del modelo
if modelo.check_model():
    print("La red bayesiana está definida correctamente.")
else:
    print("La red bayesiana contiene errores.")

# Inferencia probabilística
inferir = VariableElimination(modelo)
probabilidad_cesped_mojado = inferir.query(variables=['Cesped_Mojado'], evidence={'Lluvia': 1})
print(probabilidad_cesped_mojado)
