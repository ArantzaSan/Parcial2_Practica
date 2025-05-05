import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def reconocer_digitos_simple():
    """Realiza un reconocimiento simple de dígitos escritos a mano."""
    # Cargar el dataset de dígitos de scikit-learn
    digits = load_digits()
    X, y = digits.data, digits.target

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de regresión logística
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # Predecir las etiquetas del conjunto de prueba
    y_pred = modelo.predict(X_test)

    # Calcular la precisión del modelo
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision:.2f}")

    # Mostrar algunas predicciones
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
        plt.title(f"Predicho: {y_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    reconocer_digitos_simple()
