# 96_Backtracking.py
# Programa para investigación de vida silvestre utilizando lógica proposicional y backtracking.

# Definimos una función para verificar si una asignación de valores satisface las reglas lógicas.
def es_valida(asignacion, reglas):
    """
    Verifica si la asignación actual satisface todas las reglas lógicas.
    :param asignacion: Diccionario con las variables y sus valores asignados.
    :param reglas: Lista de funciones que representan las reglas lógicas.
    :return: True si todas las reglas son satisfechas, False en caso contrario.
    """
    for regla in reglas:
        if not regla(asignacion):
            return False
    return True

# Función principal de backtracking.
def backtracking(variables, reglas, asignacion={}):
    """
    Implementa el algoritmo de backtracking para encontrar una asignación válida.
    :param variables: Lista de variables proposicionales.
    :param reglas: Lista de funciones que representan las reglas lógicas.
    :param asignacion: Diccionario con las variables y sus valores asignados.
    :return: Una asignación válida o None si no se encuentra solución.
    """
    # Si todas las variables están asignadas, verificamos si es válida.
    if len(asignacion) == len(variables):
        if es_valida(asignacion, reglas):
            return asignacion
        return None

    # Seleccionamos la siguiente variable no asignada.
    variable = [v for v in variables if v not in asignacion][0]

    # Intentamos asignar True o False a la variable.
    for valor in [True, False]:
        asignacion[variable] = valor
        if es_valida(asignacion, reglas):
            resultado = backtracking(variables, reglas, asignacion)
            if resultado is not None:
                return resultado
        # Si no es válida, deshacemos la asignación.
        del asignacion[variable]

    # Si no se encuentra solución, retornamos None.
    return None

# Definimos las variables proposicionales relacionadas con la investigación de vida silvestre.
variables = ["hay_agua", "hay_comida", "hay_refugio", "es_habitat_adecuado"]

# Definimos las reglas lógicas como funciones.
def regla_agua_comida(asignacion):
    # Si hay agua y comida, entonces es un hábitat adecuado.
    if asignacion.get("hay_agua", False) and asignacion.get("hay_comida", False):
        return asignacion.get("es_habitat_adecuado", False)
    return True

def regla_refugio(asignacion):
    # Si hay refugio, entonces es un hábitat adecuado.
    if asignacion.get("hay_refugio", False):
        return asignacion.get("es_habitat_adecuado", False)
    return True

def regla_habitat(asignacion):
    # Si es un hábitat adecuado, debe haber al menos agua, comida o refugio.
    if asignacion.get("es_habitat_adecuado", False):
        return asignacion.get("hay_agua", False) or asignacion.get("hay_comida", False) or asignacion.get("hay_refugio", False)
    return True

# Lista de reglas.
reglas = [regla_agua_comida, regla_refugio, regla_habitat]

# Ejecutamos el algoritmo de backtracking para encontrar una asignación válida.
solucion = backtracking(variables, reglas)

# Mostramos el resultado.
if solucion:
    print("Se encontró una asignación válida:")
    for variable, valor in solucion.items():
        print(f"{variable}: {'Sí' if valor else 'No'}")
else:
    print("No se encontró una asignación válida.")
