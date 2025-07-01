import numpy as np

# 135_Arboles_Regresion_ID3.py
# Implementación de un Árbol de Regresión desde cero orientado a matemáticas
# El árbol de regresión divide los datos para minimizar el error cuadrático medio (MSE) en cada partición


# Nodo del árbol de regresión
class NodoRegresion:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Índice de la característica para dividir
        self.threshold = threshold  # Umbral de división
        self.left = left            # Subárbol izquierdo
        self.right = right          # Subárbol derecho
        self.value = value          # Valor de predicción si es hoja

# Función para calcular el error cuadrático medio (MSE)
def mse(y):
    if len(y) == 0:
        return 0
    return np.mean((y - np.mean(y)) ** 2)

# Función para encontrar la mejor división en los datos
def mejor_division(X, y):
    mejor_mse = float('inf')
    mejor_feature = None
    mejor_threshold = None

    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            izquierda = y[X[:, feature] <= threshold]
            derecha = y[X[:, feature] > threshold]
            mse_izq = mse(izquierda)
            mse_der = mse(derecha)
            mse_total = (len(izquierda) * mse_izq + len(derecha) * mse_der) / n_samples

            if mse_total < mejor_mse:
                mejor_mse = mse_total
                mejor_feature = feature
                mejor_threshold = threshold

    return mejor_feature, mejor_threshold

# Función recursiva para construir el árbol de regresión
def construir_arbol(X, y, profundidad_max=5, min_muestras=5, profundidad=0):
    # Condiciones de parada: profundidad máxima o pocas muestras
    if profundidad >= profundidad_max or len(y) <= min_muestras:
        hoja_valor = np.mean(y)
        return NodoRegresion(value=hoja_valor)

    feature, threshold = mejor_division(X, y)
    if feature is None:
        return NodoRegresion(value=np.mean(y))

    # Dividir los datos
    indices_izq = X[:, feature] <= threshold
    indices_der = X[:, feature] > threshold

    izquierda = construir_arbol(X[indices_izq], y[indices_izq], profundidad_max, min_muestras, profundidad + 1)
    derecha = construir_arbol(X[indices_der], y[indices_der], profundidad_max, min_muestras, profundidad + 1)

    return NodoRegresion(feature, threshold, izquierda, derecha)

# Función para predecir un valor usando el árbol
def predecir_uno(nodo, x):
    while nodo.value is None:
        if x[nodo.feature] <= nodo.threshold:
            nodo = nodo.left
        else:
            nodo = nodo.right
    return nodo.value

def predecir(arbol, X):
    return np.array([predecir_uno(arbol, x) for x in X])

# Ejemplo de uso matemático
if __name__ == "__main__":
    # Generamos datos sintéticos: y = 2x + ruido
    np.random.seed(0)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten() + np.random.normal(0, 1, 100)

    # Construimos el árbol de regresión
    arbol = construir_arbol(X, y, profundidad_max=3, min_muestras=5)

    # Predecimos sobre los mismos datos
    y_pred = predecir(arbol, X)

    # Mostramos el error cuadrático medio
    print("MSE:", mse(y - y_pred))

    # Graficamos resultados (opcional)
    try:
        import matplotlib.pyplot as plt
        plt.scatter(X, y, label="Datos reales")
        plt.plot(X, y_pred, color='red', label="Predicción árbol")
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib no está instalado, omitiendo gráfica.")
