import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def plot_decision_boundary(X, y, clf, title):
    """Función para graficar la frontera de decisión de un clasificador."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(Z == 1, 1, 0)  # Para contourf con etiquetas 0 y 1
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdBu, edgecolors='k')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(title)
    plt.grid(True)

# --- Caso 1: Datos Linealmente Separables ---
print("--- Caso 1: Datos Linealmente Separables ---")
X_linear, y_linear = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=0.5)
y_linear = np.where(y_linear == 0, -1, 1) # Etiquetas -1 y 1 para Perceptrón

X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42
)

# Entrenar un Perceptrón
perceptron_linear = Perceptron(max_iter=100, eta0=0.1, random_state=42)
perceptron_linear.fit(X_train_linear, y_train_linear)
y_pred_perceptron_linear = perceptron_linear.predict(X_test_linear)
accuracy_perceptron_linear = accuracy_score(y_test_linear, y_pred_perceptron_linear)
print(f"Precisión del Perceptrón: {accuracy_perceptron_linear:.2f}")

# Entrenar una Regresión Logística
logistic_linear = LogisticRegression(random_state=42)
logistic_linear.fit(X_train_linear, y_train_linear)
y_pred_logistic_linear = logistic_linear.predict(X_test_linear)
accuracy_logistic_linear = accuracy_score(y_test_linear, y_pred_logistic_linear)
print(f"Precisión de Regresión Logística: {accuracy_logistic_linear:.2f}")

# Entrenar una SVM Lineal
svm_linear = LinearSVC(random_state=42)
svm_linear.fit(X_train_linear, y_train_linear)
y_pred_svm_linear = svm_linear.predict(X_test_linear)
accuracy_svm_linear = accuracy_score(y_test_linear, y_pred_svm_linear)
print(f"Precisión de SVM Lineal: {accuracy_svm_linear:.2f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plot_decision_boundary(X_train_linear, y_train_linear, perceptron_linear, 'Perceptrón (Linealmente Separable)')
plt.scatter(X_test_linear[:, 0], X_test_linear[:, 1], c=y_test_linear, marker='x', s=50, cmap=plt.cm.RdBu)

plt.subplot(1, 3, 2)
plot_decision_boundary(X_train_linear, y_train_linear, logistic_linear, 'Regresión Logística (Linealmente Separable)')
plt.scatter(X_test_linear[:, 0], X_test_linear[:, 1], c=y_test_linear, marker='x', s=50, cmap=plt.cm.RdBu)

plt.subplot(1, 3, 3)
plot_decision_boundary(X_train_linear, y_train_linear, svm_linear, 'SVM Lineal (Linealmente Separable)')
plt.scatter(X_test_linear[:, 0], X_test_linear[:, 1], c=y_test_linear, marker='x', s=50, cmap=plt.cm.RdBu)

plt.tight_layout()
plt.show()

# --- Caso 2: Datos No Linealmente Separables ---
print("\n--- Caso 2: Datos No Linealmente Separables ---")
X_nonlinear, y_nonlinear = make_blobs(n_samples=100, centers=[[0, 0], [2, 2], [-2, 2]],
                                      random_state=42, cluster_std=1.0)
y_nonlinear = y_nonlinear % 2  # Convertir a un problema binario no lineal
y_nonlinear = np.where(y_nonlinear == 0, -1, 1)

X_train_nonlinear, X_test_nonlinear, y_train_nonlinear, y_test_nonlinear = train_test_split(
    X_nonlinear, y_nonlinear, test_size=0.3, random_state=42
)

# Entrenar un Perceptrón
perceptron_nonlinear = Perceptron(max_iter=100, eta0=0.1, random_state=42)
perceptron_nonlinear.fit(X_train_nonlinear, y_train_nonlinear)
y_pred_perceptron_nonlinear = perceptron_nonlinear.predict(X_test_nonlinear)
accuracy_perceptron_nonlinear = accuracy_score(y_test_nonlinear, y_pred_perceptron_nonlinear)
print(f"Precisión del Perceptrón: {accuracy_perceptron_nonlinear:.2f}")

# Entrenar una Regresión Logística
logistic_nonlinear = LogisticRegression(random_state=42)
logistic_nonlinear.fit(X_train_nonlinear, y_train_nonlinear)
y_pred_logistic_nonlinear = logistic_nonlinear.predict(X_test_nonlinear)
accuracy_logistic_nonlinear = accuracy_score(y_test_nonlinear, y_pred_logistic_nonlinear)
print(f"Precisión de Regresión Logística: {accuracy_logistic_nonlinear:.2f}")

# Entrenar una SVM Lineal
svm_nonlinear = LinearSVC(random_state=42)
svm_nonlinear.fit(X_train_nonlinear, y_train_nonlinear)
y_pred_svm_nonlinear = svm_nonlinear.predict(X_test_nonlinear)
accuracy_svm_nonlinear = accuracy_score(y_test_nonlinear, y_pred_svm_nonlinear)
print(f"Precisión de SVM Lineal: {accuracy_svm_nonlinear:.2f}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plot_decision_boundary(X_train_nonlinear, y_train_nonlinear, perceptron_nonlinear, 'Perceptrón (No Linealmente Separable)')
plt.scatter(X_test_nonlinear[:, 0], X_test_nonlinear[:, 1], c=y_test_nonlinear, marker='x', s=50, cmap=plt.cm.RdBu)

plt.subplot(1, 3, 2)
plot_decision_boundary(X_train_nonlinear, y_train_nonlinear, logistic_nonlinear, 'Regresión Logística (No Linealmente Separable)')
plt.scatter(X_test_nonlinear[:, 0], X_test_nonlinear[:, 1], c=y_test_nonlinear, marker='x', s=5
