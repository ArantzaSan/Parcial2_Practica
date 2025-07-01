import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 136_Conjunto_Hipoteis_Boosting.py
# Implementación de conjunto de hipótesis y boosting orientado a biología
# Ejemplo: Clasificación de flores según características (tipo Iris)


# ==========================
# 1. Cargar y preparar datos
# ==========================
# Usamos el dataset Iris, común en biología para clasificación de especies de flores
iris = load_iris()
X = iris.data  # Características: largo/peso de pétalos y sépalos
y = iris.target  # Clases: especie de flor

# Para simplificar, seleccionamos solo dos clases (binario)
X = X[y != 2]
y = y[y != 2]

# ==========================
# 2. Inicializar pesos
# ==========================
# Inicializamos los pesos de cada muestra de manera uniforme
n_samples = X.shape[0]
weights = np.ones(n_samples) / n_samples

# ==========================
# 3. Boosting: AdaBoost simple
# ==========================
n_estimators = 5  # Número de hipótesis débiles (árboles pequeños)
estimators = []
alphas = []

for i in range(n_estimators):
    # Entrenamos un árbol de decisión débil (stump)
    stump = DecisionTreeClassifier(max_depth=1, random_state=i)
    stump.fit(X, y, sample_weight=weights)
    pred = stump.predict(X)
    
    # Calculamos el error ponderado
    miss = (pred != y)
    error = np.dot(weights, miss) / np.sum(weights)
    
    # Evitar división por cero
    if error == 0:
        alpha = 1
    else:
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
    
    # Guardamos el clasificador y su peso
    estimators.append(stump)
    alphas.append(alpha)
    
    # Actualizamos los pesos: más peso a los errores
    weights *= np.exp(-alpha * y * (2 * pred - 1))
    weights /= np.sum(weights)  # Normalizamos

# ==========================
# 4. Conjunto de hipótesis
# ==========================
# La predicción final es una combinación ponderada de las hipótesis débiles
def predict_boosting(X):
    final_pred = np.zeros(X.shape[0])
    for alpha, est in zip(alphas, estimators):
        pred = est.predict(X)
        # Convertimos las clases {0,1} a {-1,1}
        pred = 2 * pred - 1
        final_pred += alpha * pred
    # Decisión final: signo de la suma ponderada
    return (final_pred > 0).astype(int)

# ==========================
# 5. Evaluación
# ==========================
y_pred = predict_boosting(X)
acc = accuracy_score(y, y_pred)
print(f"Precisión del conjunto de hipótesis (boosting): {acc:.2f}")

# ==========================
# 6. Comentarios finales
# ==========================
# Este programa implementa boosting (AdaBoost) usando árboles de decisión débiles
# para clasificar especies de flores (Iris). El conjunto de hipótesis mejora la
# precisión combinando varios clasificadores simples, útil en problemas biológicos
# donde los datos pueden ser complejos o ruidosos.
