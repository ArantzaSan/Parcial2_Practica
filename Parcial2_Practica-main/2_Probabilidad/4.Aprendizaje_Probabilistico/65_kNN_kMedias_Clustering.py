import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter
from scipy.spatial.distance import euclidean

# --- k-Nearest Neighbors (kNN) para Clasificación (aunque el título pide clustering) ---
# Nota: kNN es un algoritmo de clasificación supervisado, no de clustering no supervisado.
# Lo incluyo aquí porque el título lo menciona junto con k-Medias.

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcular distancias entre x y todos los puntos de entrenamiento
        distances = [euclidean(x, x_train) for x_train in self.X_train]
        # Obtener los índices de los k vecinos más cercanos
        k_nearest_indices = np.argsort(distances)[:self.k]
        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        # Devolver la etiqueta más común entre los vecinos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# --- k-Means Clustering (k-Medias) ---

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Inicializar centroides aleatoriamente
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Asignar cada punto al centroide más cercano
            distances = np.array([[euclidean(x, centroid) for centroid in self.centroids] for x in X])
            self.labels = np.argmin(distances, axis=1)

            # Actualizar los centroides calculando la media de los puntos asignados a cada cluster
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Verificar convergencia
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        return self

    def predict(self, X):
        distances = np.array([[euclidean(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Generar datos de ejemplo para clustering
    X_clust, y_clust_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    # --- k-Means Clustering ---
    kmeans = KMeansClustering(n_clusters=3, random_state=0)
    kmeans.fit(X_clust)
    labels_kmeans = kmeans.predict(X_clust)
    centroids_kmeans = kmeans.centroids

    print("--- k-Means Clustering ---")
    print("Centroides encontrados:")
    print(centroids_kmeans)

    # Visualizar resultados de k-Means
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_clust[:, 0], X_clust[:, 1], c=labels_kmeans, cmap='viridis', s=50)
    plt.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1], marker='X', s=200, color='red', label='Centroides')
    plt.title('k-Means Clustering')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)

    # --- k-Nearest Neighbors (para clasificación usando las etiquetas verdaderas) ---
    # Nota: Esto es solo para demostrar kNN. Para un problema de clustering real, no tendrías y_clust_true.
    X_knn, y_knn = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)

    knn_classifier = KNNClassifier(k=3)
    knn_classifier.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn_classifier.predict(X_test_knn)

    accuracy_knn = np.mean(y_pred_knn == y_test_knn)
    print("\n--- k-Nearest Neighbors (Clasificación) ---")
    print(f"Precisión de kNN en datos de clasificación: {accuracy_knn:.2f}")

    plt.subplot(1, 2, 2)
    plt.scatter(X_test_knn[:, 0], X_test_knn[:, 1], c=y_pred_knn, cmap='plasma', s=50, label='Predicciones kNN')
    plt.scatter(X_train_knn[:, 0], X_train_knn[:, 1], c=y_train_knn, cmap='plasma', s=50, alpha=0.3, label='Datos de Entrenamiento')
    plt.title('k-Nearest Neighbors (Clasificación)')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
