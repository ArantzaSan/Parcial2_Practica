import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-3, random_state=None):
        """
        Modelo de Mezcla Gaussiana (Gaussian Mixture Model - GMM) utilizando el algoritmo EM.

        Args:
            n_components (int): El número de componentes (clusters) gaussianos a ajustar.
            max_iter (int): El número máximo de iteraciones del algoritmo EM.
            tol (float): Tolerancia para la convergencia del algoritmo EM.
            random_state (int): Semilla para la generación de números aleatorios para la inicialización.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.responsibilities_ = None
        self.converged_ = False
        self.n_iter_ = 0

    def _initialize(self, X):
        """Inicializa los parámetros del modelo."""
        n_samples, n_features = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Inicializar pesos uniformemente
        self.weights_ = np.ones(self.n_components) / self.n_components

        # Inicializar medias aleatoriamente a partir de los datos
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_indices]

        # Inicializar covarianzas como matrices identidad escaladas
        self.covariances_ = [np.eye(n_features) for _ in range(self.n_components)]

    def _e_step(self, X):
        """Paso E: Calcular las responsabilidades."""
        n_samples = X.shape[0]
        self.responsibilities_ = np.zeros((n_samples, self.n_components))
        for i in range(self.n_components):
            numerator = self.weights_[i] * self._gaussian_pdf(X, self.means_[i], self.covariances_[i])
            self.responsibilities_[:, i] = numerator

        # Normalizar las responsabilidades
        denominator = np.sum(self.responsibilities_, axis=1, keepdims=True)
        self.responsibilities_ /= denominator + 1e-9  # Evitar división por cero

    def _m_step(self, X):
        """Paso M: Actualizar los parámetros."""
        n_samples = X.shape[0]
        Nk = np.sum(self.responsibilities_, axis=0)

        # Actualizar pesos
        self.weights_ = Nk / n_samples

        # Actualizar medias
        self.means_ = np.zeros_like(self.means_)
        for i in range(self.n_components):
            self.means_[i] = np.sum(self.responsibilities_[:, i][:, np.newaxis] * X, axis=0) / Nk[i]

        # Actualizar covarianzas
        self.covariances_ = []
        for i in range(self.n_components):
            diff = X - self.means_[i]
            weighted_diff = self.responsibilities_[:, i][:, np.newaxis] * diff
            covariance = (weighted_diff.T @ diff) / Nk[i]
            # Asegurar que la matriz de covarianza sea definida positiva
            covariance += np.eye(X.shape[1]) * 1e-6
            self.covariances_.append(covariance)

    def _gaussian_pdf(self, X, mean, covariance):
        """Calcula la función de densidad de probabilidad gaussiana multivariante."""
        n_features = X.shape[1]
        diff = X - mean
        try:
            inv_cov = np.linalg.inv(covariance)
            det_cov = np.linalg.det(covariance)
            numerator = np.exp(-0.5 * np.sum(diff @ inv_cov @ diff.T, axis=1))
            denominator = np.sqrt((2 * np.pi) ** n_features * det_cov)
            return numerator / denominator
        except np.linalg.LinAlgError:
            return np.zeros(X.shape[0])

    def fit(self, X):
        """Ajusta el modelo a los datos."""
        self._initialize(X)
        prev_log_likelihood = -np.inf
        for self.n_iter_ in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)

            # Calcular el log-likelihood para verificar la convergencia
            log_likelihood = self.log_likelihood(X)
            if log_likelihood - prev_log_likelihood < self.tol:
                self.converged_ = True
                break
            prev_log_likelihood = log_likelihood

        return self

    def predict(self, X):
        """Predice la asignación de cluster para cada muestra."""
        self._e_step(X)
        return np.argmax(self.responsibilities_, axis=1)

    def predict_proba(self, X):
        """Devuelve las probabilidades de pertenencia a cada cluster."""
        self._e_step(X)
        return self.responsibilities_

    def log_likelihood(self, X):
        """Calcula el log-likelihood de los datos bajo el modelo ajustado."""
        n_samples = X.shape[0]
        log_likelihood = 0
        for i in range(n_samples):
            likelihood_sample = 0
            for j in range(self.n_components):
                likelihood_sample += self.weights_[j] * self._gaussian_pdf(X[i], self.means_[j], self.covariances_[j])
            log_likelihood += np.log(likelihood_sample + 1e-9)
        return log_likelihood

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo para clustering
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    # Crear e entrenar el modelo GMM
    n_clusters = 3
    gmm = GaussianMixtureModel(n_components=n_clusters, max_iter=200, random_state=0)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    print("Medias de los clusters:")
    print(gmm.means_)
    print("\nCovarianzas de los clusters:")
    print(gmm.covariances_)
    print("\nPesos de los componentes:")
    print(gmm.weights_)
    print("\nLog-likelihood del modelo:")
    print(gmm.log_likelihood(X))

    # Visualizar los resultados del clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='X', s=200, color='red', label='Centros de los Clusters')

    # Dibujar las elipses de covarianza (solo para 2D)
    if X.shape[1] == 2:
        from matplotlib.patches import Ellipse
        ax = plt.gca()
        for i in range(n_clusters):
            eigenvalues, eigenvectors = np.linalg.eig(gmm.covariances_[i])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=gmm.means_[i], width=width, height=height, angle=angle, alpha=0.3, color='red')
            ax.add_patch(ellipse)

    plt.title('Agrupamiento con Modelo de Mezcla Gaussiana (GMM)')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)
    plt.show()
