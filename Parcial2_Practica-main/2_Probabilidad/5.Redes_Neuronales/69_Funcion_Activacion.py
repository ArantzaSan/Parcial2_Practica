import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-x))

def tangente_hiperbolica(x):
    """Función de activación tangente hiperbólica (tanh)."""
    return np.tanh(x)

def relu(x):
    """Función de activación Rectified Linear Unit (ReLU)."""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Función de activación Leaky ReLU."""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """Función de activación Exponential Linear Unit (ELU)."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    """Función de activación softmax (para clasificación multiclase)."""
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# Generar valores de entrada para graficar
x = np.linspace(-5, 5, 100)

# Calcular las salidas de cada función de activación
y_sigmoide = sigmoide(x)
y_tanh = tangente_hiperbolica(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)

# Ejemplo de entrada para softmax (para demostrar su comportamiento)
x_softmax = np.array([1.0, 2.0, 0.5])
y_softmax = softmax(x_softmax)
print(f"Entrada para Softmax: {x_softmax}")
print(f"Salida de Softmax: {y_softmax} (Suma: {np.sum(y_softmax):.2f})")

# Graficar las funciones de activación
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, y_sigmoide)
plt.title('Sigmoide')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, y_tanh)
plt.title('Tangente Hiperbólica (tanh)')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, y_relu)
plt.title('ReLU')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, y_leaky_relu)
plt.title('Leaky ReLU')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, y_elu)
plt.title('ELU')
plt.xlabel('Entrada')
plt.ylabel('Salida')
plt.grid(True)

plt.tight_layout()
plt.show()

# Consideraciones sobre las Funciones de Activación:

print("\nConsideraciones sobre las Funciones de Activación:")
print("- **Sigmoide:** Salida entre 0 y 1, históricamente popular pero sufre de gradientes que se desvanecen en redes profundas.")
print("- **Tangente Hiperbólica (tanh):** Salida entre -1 y 1, similar al sigmoide pero centrada en cero, lo que a veces ayuda en el entrenamiento.")
print("- **ReLU:** Simple y computacionalmente eficiente, no sufre del problema de gradientes desvanecientes para entradas positivas, pero puede sufrir de la 'neurona muerta' (salida siempre cero para entradas negativas).")
print("- **Leaky ReLU:** Intenta solucionar el problema de la neurona muerta introduciendo una pequeña pendiente para entradas negativas.")
print("- **ELU:** Similar a ReLU para entradas positivas pero suaviza la salida para entradas negativas hacia -alpha, lo que puede ayudar en el aprendizaje.")
print("- **Softmax:** Se utiliza típicamente en la capa de
