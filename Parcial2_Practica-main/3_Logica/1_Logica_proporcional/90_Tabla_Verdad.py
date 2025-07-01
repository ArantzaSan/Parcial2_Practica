import itertools

# Importamos la biblioteca itertools para generar combinaciones de valores de verdad

# Definimos las variables de entrada para la tabla de verdad
# En electrónica, estas podrían representar señales de entrada (A, B, etc.)
variables = ['A', 'B']

# Definimos las funciones lógicas que queremos evaluar
# Estas funciones representan compuertas lógicas comunes en electrónica
def AND(a, b):
    return a and b

def OR(a, b):
    return a or b

def NOT(a):
    return not a

def XOR(a, b):
    return a ^ b

# Generamos todas las combinaciones posibles de valores de verdad para las variables
# Cada variable puede ser True (1) o False (0)
combinaciones = list(itertools.product([False, True], repeat=len(variables)))

# Imprimimos el encabezado de la tabla de verdad
print(f"{'A':<5}{'B':<5}{'A AND B':<10}{'A OR B':<10}{'A XOR B':<10}{'NOT A':<10}")
print("-" * 50)

# Iteramos sobre cada combinación de valores de verdad
for combinacion in combinaciones:
    A, B = combinacion  # Asignamos los valores de la combinación a las variables
    # Calculamos los resultados de las funciones lógicas
    resultado_and = AND(A, B)
    resultado_or = OR(A, B)
    resultado_xor = XOR(A, B)
    resultado_not_a = NOT(A)
    
    # Imprimimos los resultados en formato de tabla
    print(f"{A!s:<5}{B!s:<5}{resultado_and!s:<10}{resultado_or!s:<10}{resultado_xor!s:<10}{resultado_not_a!s:<10}")
