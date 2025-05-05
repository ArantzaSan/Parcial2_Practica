import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean

class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)  # Convertir etiquetas a -1 y 1

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo para clasificación binaria
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y = np.where(y == 0, -1, 1)  # Etiquetas -1 y 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Entrenar el clasificador SVM
    svm = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)

    # Calcular la precisión
    accuracy = np.mean(predictions == y_test)
    print(f"Precisión del SVM: {accuracy:.2f}")

    # Visualizar la frontera de decisión (para datos bidimensionales)
    def plot_decision_boundary(X, y, clf):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.title('Frontera de Decisión del SVM')

    plt.figure(figsize=(8, 6))
    plot_decision_boundary(X_train, y_train, svm)
    plt.show()
