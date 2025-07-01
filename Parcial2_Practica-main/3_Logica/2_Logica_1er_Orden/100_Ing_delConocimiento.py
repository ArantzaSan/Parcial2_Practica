from sympy.logic.boolalg import And, Or, Not, Implies
from sympy.abc import A, B, C, D, E
from sympy.logic.inference import satisfiable

# Importamos la biblioteca `sympy` que nos ayudará a trabajar con lógica simbólica

# Definimos las proposiciones básicas que representan hechos médicos
# A: El paciente tiene fiebre
# B: El paciente tiene tos
# C: El paciente tiene dificultad para respirar
# D: El paciente tiene infección viral
# E: El paciente tiene neumonía

# 1. Definimos las reglas de conocimiento médico usando lógica de primer orden
# Regla 1: Si el paciente tiene fiebre y tos, entonces podría tener una infección viral
regla_1 = Implies(And(A, B), D)

# Regla 2: Si el paciente tiene fiebre, tos y dificultad para respirar, entonces podría tener neumonía
regla_2 = Implies(And(A, B, C), E)

# Regla 3: Si el paciente tiene neumonía, entonces también tiene una infección viral
regla_3 = Implies(E, D)

# 2. Definimos los hechos conocidos sobre el paciente
# Por ejemplo, sabemos que el paciente tiene fiebre y tos
hecho_1 = A  # El paciente tiene fiebre
hecho_2 = B  # El paciente tiene tos

# 3. Inferimos nuevos hechos basados en las reglas y los hechos conocidos
# Usamos las reglas para deducir si el paciente tiene una infección viral o neumonía

# Verificamos si el paciente tiene una infección viral
infeccion_viral = satisfiable(And(hecho_1, hecho_2, regla_1))
print("¿El paciente tiene una infección viral?", infeccion_viral)

# Verificamos si el paciente tiene neumonía
neumonia = satisfiable(And(hecho_1, hecho_2, regla_2))
print("¿El paciente tiene neumonía?", neumonia)

# 4. Explicamos los resultados
# Basándonos en las reglas y hechos, el programa deduce si el paciente tiene una infección viral o neumonía.
# Este enfoque puede extenderse agregando más reglas y hechos para cubrir otros diagnósticos médicos.
