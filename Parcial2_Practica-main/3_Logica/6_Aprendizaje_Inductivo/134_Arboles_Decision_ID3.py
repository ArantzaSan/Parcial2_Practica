import math
from collections import Counter

# 134_Arboles_Decision_ID3.py
# Árbol de Decisión ID3 aplicado a un ejemplo de electrónica
# Ejemplo: Diagnóstico de fallas en un circuito simple


# -------------------------------
# 1. Definición del conjunto de datos
# Cada ejemplo representa un circuito con ciertas características y si funciona o no
# Atributos: [voltaje_correcto, resistencia_buena, conexion_firme, funciona]
# Los valores son 'si' o 'no'
datos = [
    {'voltaje_correcto': 'si', 'resistencia_buena': 'si', 'conexion_firme': 'si', 'funciona': 'si'},
    {'voltaje_correcto': 'no', 'resistencia_buena': 'si', 'conexion_firme': 'si', 'funciona': 'no'},
    {'voltaje_correcto': 'si', 'resistencia_buena': 'no', 'conexion_firme': 'si', 'funciona': 'no'},
    {'voltaje_correcto': 'si', 'resistencia_buena': 'si', 'conexion_firme': 'no', 'funciona': 'no'},
    {'voltaje_correcto': 'no', 'resistencia_buena': 'no', 'conexion_firme': 'si', 'funciona': 'no'},
    {'voltaje_correcto': 'no', 'resistencia_buena': 'si', 'conexion_firme': 'no', 'funciona': 'no'},
    {'voltaje_correcto': 'si', 'resistencia_buena': 'no', 'conexion_firme': 'no', 'funciona': 'no'},
    {'voltaje_correcto': 'no', 'resistencia_buena': 'no', 'conexion_firme': 'no', 'funciona': 'no'},
    {'voltaje_correcto': 'si', 'resistencia_buena': 'si', 'conexion_firme': 'si', 'funciona': 'si'},
]

atributos = ['voltaje_correcto', 'resistencia_buena', 'conexion_firme']

# -------------------------------
# 2. Función para calcular la entropía de un conjunto de datos
def entropia(datos, objetivo):
    total = len(datos)
    conteo = Counter([fila[objetivo] for fila in datos])
    return -sum((n/total) * math.log2(n/total) for n in conteo.values() if n > 0)

# -------------------------------
# 3. Función para calcular la ganancia de información de un atributo
def ganancia_informacion(datos, atributo, objetivo):
    total = len(datos)
    valores = set(fila[atributo] for fila in datos)
    entropia_total = entropia(datos, objetivo)
    entropia_condicional = 0
    for valor in valores:
        subconjunto = [fila for fila in datos if fila[atributo] == valor]
        peso = len(subconjunto) / total
        entropia_condicional += peso * entropia(subconjunto, objetivo)
    return entropia_total - entropia_condicional

# -------------------------------
# 4. Función recursiva para construir el árbol de decisión
def id3(datos, atributos, objetivo):
    # Si todos los ejemplos tienen la misma clase, retornar esa clase
    clases = set(fila[objetivo] for fila in datos)
    if len(clases) == 1:
        return next(iter(clases))
    # Si no hay atributos restantes, retornar la clase mayoritaria
    if not atributos:
        return Counter([fila[objetivo] for fila in datos]).most_common(1)[0][0]
    # Elegir el mejor atributo según la ganancia de información
    mejor = max(atributos, key=lambda a: ganancia_informacion(datos, a, objetivo))
    arbol = {mejor: {}}
    for valor in set(fila[mejor] for fila in datos):
        subconjunto = [fila for fila in datos if fila[mejor] == valor]
        if subconjunto:
            arbol[mejor][valor] = id3(
                subconjunto,
                [a for a in atributos if a != mejor],
                objetivo
            )
        else:
            # Si no hay ejemplos, retornar la clase mayoritaria
            arbol[mejor][valor] = Counter([fila[objetivo] for fila in datos]).most_common(1)[0][0]
    return arbol

# -------------------------------
# 5. Construcción del árbol de decisión
arbol = id3(datos, atributos, 'funciona')

# -------------------------------
# 6. Función para predecir con el árbol construido
def predecir(arbol, ejemplo):
    if isinstance(arbol, str):
        return arbol
    atributo = next(iter(arbol))
    valor = ejemplo.get(atributo)
    if valor in arbol[atributo]:
        return predecir(arbol[atributo][valor], ejemplo)
    else:
        return 'Desconocido'

# -------------------------------
# 7. Ejemplo de uso
ejemplo_nuevo = {'voltaje_correcto': 'si', 'resistencia_buena': 'si', 'conexion_firme': 'no'}
print("Árbol de decisión:", arbol)
print("Predicción para el ejemplo nuevo:", predecir(arbol, ejemplo_nuevo))

# -------------------------------
# Fin del programa
