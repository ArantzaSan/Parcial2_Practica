from sympy import symbols
from sympy.logic.boolalg import Implies, And, Or, Not, satisfiable

# Importamos la biblioteca sympy para trabajar con lógica proposicional

# Definimos las proposiciones lógicas
# Por ejemplo, "P" puede representar "El acusado es culpable"
# y "Q" puede representar "Hay pruebas suficientes"
P = symbols('P')
Q = symbols('Q')

# Definimos las reglas de inferencia
# Por ejemplo, si "P implica Q" (si el acusado es culpable, entonces hay pruebas suficientes)
regla1 = Implies(P, Q)

# Definimos un caso hipotético
# Por ejemplo, "No hay pruebas suficientes" (Not(Q))
caso = Not(Q)

# Evaluamos si el caso es consistente con las reglas de inferencia
# Esto nos ayuda a determinar si hay una contradicción lógica
resultado = satisfiable(And(regla1, caso))

# Analizamos el resultado
# Si el resultado es False, hay una contradicción (falacia)
# Si el resultado es un diccionario, el caso es consistente (verdad)
if resultado is False:
    print("El caso contiene una falacia lógica.")
else:
    print("El caso es consistente con las reglas de inferencia.")
    print("Asignaciones posibles:", resultado)

# Ejemplo adicional: verificamos si una conclusión es válida
# Por ejemplo, "El acusado no es culpable" (Not(P))
conclusion = Not(P)

# Evaluamos si la conclusión es válida bajo las reglas y el caso
validez = satisfiable(And(regla1, caso, Not(conclusion)))

if validez is False:
    print("La conclusión es válida.")
else:
    print("La conclusión no es válida.")
    print("Asignaciones posibles:", validez)
