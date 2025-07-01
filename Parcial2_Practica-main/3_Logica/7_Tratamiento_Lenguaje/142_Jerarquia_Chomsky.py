# 142_Jerarquia_Chomsky.py
# Implementación básica de la Jerarquía de Chomsky en Python

# Definimos una clase base para Gramática
class Gramatica:
    def __init__(self, variables, terminales, producciones, inicial):
        self.variables = variables          # No terminales
        self.terminales = terminales        # Terminales
        self.producciones = producciones    # Diccionario: variable -> lista de producciones
        self.inicial = inicial              # Símbolo inicial

# Gramática Tipo 0: Gramática irrestricta
class GramaticaTipo0(Gramatica):
    def tipo(self):
        return "Tipo 0: Gramática irrestricta (Turing-completa)"

# Gramática Tipo 1: Sensible al contexto
class GramaticaTipo1(Gramatica):
    def tipo(self):
        return "Tipo 1: Sensible al contexto"

# Gramática Tipo 2: Libre de contexto
class GramaticaTipo2(Gramatica):
    def tipo(self):
        return "Tipo 2: Libre de contexto"

# Gramática Tipo 3: Regular
class GramaticaTipo3(Gramatica):
    def tipo(self):
        return "Tipo 3: Regular"

# Ejemplo de uso: definimos una gramática regular (Tipo 3)
variables = {'S', 'A'}
terminales = {'a', 'b'}
producciones = {
    'S': ['aA', 'b'],
    'A': ['a', 'bS']
}
inicial = 'S'

# Creamos una instancia de la gramática regular
gramatica = GramaticaTipo3(variables, terminales, producciones, inicial)

# Mostramos información de la gramática
print("Variables:", gramatica.variables)
print("Terminales:", gramatica.terminales)
print("Producciones:")
for var, prods in gramatica.producciones.items():
    for prod in prods:
        print(f"  {var} -> {prod}")
print("Símbolo inicial:", gramatica.inicial)
print("Tipo de gramática:", gramatica.tipo())

# Nota: Para una implementación completa, se pueden agregar métodos para verificar
# si una gramática pertenece a un tipo específico según las reglas de la jerarquía.
