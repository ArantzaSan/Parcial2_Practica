import math

# Importamos la librería math para realizar cálculos matemáticos

# Definimos una función de orden superior que toma una función y una lista de puntos
def aplicar_a_puntos(funcion, puntos):
    """
    Aplica una función a una lista de puntos.
    :param funcion: Función a aplicar (debe aceptar un punto como argumento).
    :param puntos: Lista de puntos (cada punto es una tupla (x, y)).
    :return: Lista de resultados de aplicar la función a cada punto.
    """
    return [funcion(punto) for punto in puntos]

# Función para calcular la distancia de un punto al origen
def distancia_al_origen(punto):
    """
    Calcula la distancia de un punto al origen (0, 0).
    :param punto: Tupla (x, y) que representa un punto en el plano.
    :return: Distancia al origen.
    """
    x, y = punto
    return math.sqrt(x**2 + y**2)

# Función para calcular el área de un triángulo dado tres puntos
def area_triangulo(p1, p2, p3):
    """
    Calcula el área de un triángulo dados tres puntos en el plano.
    :param p1: Primer punto (x1, y1).
    :param p2: Segundo punto (x2, y2).
    :param p3: Tercer punto (x3, y3).
    :return: Área del triángulo.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2)

# Lista de puntos en el plano
puntos = [(1, 2), (3, 4), (5, 6), (7, 8)]

# Aplicamos la función distancia_al_origen a cada punto usando la función de orden superior
distancias = aplicar_a_puntos(distancia_al_origen, puntos)

# Mostramos las distancias calculadas
print("Distancias al origen:", distancias)

# Calculamos el área de un triángulo formado por tres puntos específicos
p1, p2, p3 = (0, 0), (4, 0), (0, 3)
area = area_triangulo(p1, p2, p3)

# Mostramos el área del triángulo
print("Área del triángulo:", area)
