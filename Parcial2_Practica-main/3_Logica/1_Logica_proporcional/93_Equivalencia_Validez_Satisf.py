from sympy import symbols, Or, And, Not, Implies, Equivalent, satisfiable

# Importamos la librería sympy para trabajar con lógica proposicional

# Definimos las proposiciones básicas
# p: El producto pasa la inspección visual
# q: El producto pasa la inspección funcional
# r: El producto cumple con las especificaciones de diseño
p, q, r = symbols('p q r')

# 1. EQUIVALENCIA
# Verificamos si dos expresiones lógicas son equivalentes
# Ejemplo: Un producto es aceptable si pasa la inspección visual y funcional
expr1 = And(p, q)  # Inspección visual y funcional
expr2 = Not(Or(Not(p), Not(q)))  # Negación de que falle alguna inspección
equivalencia = Equivalent(expr1, expr2)
print(f"¿Las expresiones son equivalentes? {equivalencia}")

# 2. VALIDEZ
# Una expresión es válida si es verdadera en todos los casos posibles
# Ejemplo: Si un producto cumple con las especificaciones de diseño, entonces pasa la inspección funcional
expr3 = Implies(r, q)  # Si cumple diseño, pasa inspección funcional
validez = expr3.simplify()  # Simplificamos para verificar validez
print(f"¿La expresión es válida? {validez}")

# 3. SATISFACCIÓN
# Una expresión es satisfactoria si existe al menos un caso donde sea verdadera
# Ejemplo: Un producto pasa al menos una de las inspecciones
expr4 = Or(p, q)  # Pasa inspección visual o funcional
satisfaccion = satisfiable(expr4)
print(f"¿La expresión es satisfactoria? {satisfaccion}")

# Aplicación en la manufacturera
# Evaluamos un caso práctico: Un producto debe pasar todas las inspecciones para ser aceptado
producto_aceptado = And(p, q, r)  # Pasa visual, funcional y cumple diseño
es_satisfactorio = satisfiable(producto_aceptado)
print(f"¿El producto puede ser aceptado? {es_satisfactorio}")

# Conclusión
# Este programa permite verificar equivalencias, validez y satisfacción en lógica proposicional,
# aplicado al control de calidad de una manufacturera.
