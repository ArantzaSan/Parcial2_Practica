import re

# 145_Analisis_Semantico.py

# Importamos las librerías necesarias

# Definimos una clase para el análisis semántico
class AnalizadorSemantico:
    def __init__(self):
        # Definimos un diccionario de variables y sus tipos
        self.tabla_simbolos = {}

    # Método para analizar una línea de código simple (asignaciones)
    def analizar_linea(self, linea):
        # Expresión regular para detectar asignaciones: variable = valor
        patron = r'^\s*(\w+)\s*=\s*(.+)$'
        match = re.match(patron, linea)
        if match:
            variable = match.group(1)
            valor = match.group(2).strip()
            tipo = self.inferir_tipo(valor)
            # Verificamos si la variable ya existe y si el tipo es consistente
            if variable in self.tabla_simbolos:
                if self.tabla_simbolos[variable] != tipo:
                    print(f"Error semántico: Tipo inconsistente para '{variable}'. Esperado {self.tabla_simbolos[variable]}, encontrado {tipo}.")
            else:
                self.tabla_simbolos[variable] = tipo
        else:
            print("Línea no reconocida para análisis semántico.")

    # Método para inferir el tipo de un valor
    def inferir_tipo(self, valor):
        if re.match(r'^-?\d+$', valor):
            return 'int'
        elif re.match(r'^-?\d+\.\d+$', valor):
            return 'float'
        elif re.match(r'^".*"$', valor) or re.match(r"^'.*'$", valor):
            return 'str'
        elif valor in ['True', 'False']:
            return 'bool'
        else:
            # Si es una variable ya declarada
            if valor in self.tabla_simbolos:
                return self.tabla_simbolos[valor]
            return 'desconocido'

    # Método para mostrar la tabla de símbolos
    def mostrar_tabla(self):
        print("Tabla de símbolos:")
        for var, tipo in self.tabla_simbolos.items():
            print(f"  {var}: {tipo}")

# Sección principal del programa
if __name__ == "__main__":
    # Creamos una instancia del analizador semántico
    analizador = AnalizadorSemantico()

    # Lista de líneas de código a analizar
    lineas = [
        "x = 5",
        "y = 3.14",
        "nombre = 'Juan'",
        "activo = True",
        "x = 'texto'",  # Esto debe dar error semántico
        "z = x"
    ]

    # Analizamos cada línea
    for linea in lineas:
        print(f"Analizando: {linea}")
        analizador.analizar_linea(linea)

    # Mostramos la tabla de símbolos final
    analizador.mostrar_tabla()
