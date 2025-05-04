from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Crear la estructura de la red bayesiana
model = BayesianNetwork([('Lluvia', 'Charcos'), 
                         ('Aspersor', 'Charcos'), 
                         ('Lluvia', 'Cesped_Mojado'), 
                         ('Charcos', 'Cesped_Mojado')])

# Definir las distribuciones de probabilidad condicional (CPDs)
cpd_lluvia = TabularCPD(variable='Lluvia', variable_card=2, values=[[0.7], [0.3]])
cpd_aspersor = TabularCPD(variable='Aspersor', variable_card=2, values=[[0.6], [0.4]])
cpd_charcos = TabularCPD(variable='Charcos', variable_card=2,
                         values=[[0.9, 0.4, 0.3, 0.1],
                                 [0.1, 0.6, 0.7, 0.9]],
                         evidence=['Lluvia', 'Aspersor'], evidence_card=[2, 2])
cpd_cesped_mojado = TabularCPD(variable='Cesped_Mojado', variable_card=2,
                               values=[[0.8, 0.2, 0.1, 0.05],
                                       [0.2, 0.8, 0.9, 0.95]],
                               evidence=['Lluvia', 'Charcos'], evidence_card=[2, 2])

# Añadir las CPDs al modelo
model.add_cpds(cpd_lluvia, cpd_aspersor, cpd_charcos, cpd_cesped_mojado)

# Verificar si la red está correctamente definida
if model.check_model():
    print("La red bayesiana está correctamente definida.")
else:
    print("La red bayesiana tiene errores.")

# Ejemplo de inferencia

inference = VariableElimination(model)
prob_cesped_mojado = inference.query(variables=['Cesped_Mojado'], evidence={'Lluvia': 1})
print(prob_cesped_mojado)