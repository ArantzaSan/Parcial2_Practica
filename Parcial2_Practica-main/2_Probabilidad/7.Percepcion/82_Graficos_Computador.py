import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def graficar_punto(x, y, color='red', etiqueta=None):
    """Grafica un punto 2D."""
    plt.plot(x, y, marker='o', linestyle='', color=color, label=etiqueta)
    if etiqueta:
        plt.legend()
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.title("Gráfico de Punto 2D")
    plt.grid(True)
    plt.show()

def graficar_linea(x, y, color='blue', etiqueta=None):
    """Grafica una línea 2D."""
    plt.plot(x, y, color=color, label=etiqueta)
    if etiqueta:
        plt.legend()
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.title("Gráfico de Línea 2D")
    plt.grid(True)
    plt.show()

def graficar_dispersion(x, y, color='green', etiqueta=None):
    """Grafica un diagrama de dispersión 2D."""
    plt.scatter(x, y, color=color, label=etiqueta)
    if etiqueta:
        plt.legend()
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.title("Diagrama de Dispersión 2D")
    plt.grid(True)
    plt.show()

def graficar_superficie_3d():
    """Grafica una superficie 3D simple."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num1 = np.arange(-5, 5, 0.25)
    num2 = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(num1, num2)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel("Eje X")
    ax.set_ylabel("Eje Y")
    ax.set_zlabel("Eje Z")
    ax.set_title("Superficie 3D")
    plt.show()

if __name__ == "__main__":
    graficar_punto(2, 3, etiqueta="Punto A")
    graficar_linea([1, 2, 3, 4], [2, 4, 1, 3], color='purple', etiqueta="Línea B")
    graficar_dispersion(np.random.rand(50), np.random.rand(50), color='orange', etiqueta="Datos C")
    graficar_superficie_3d()
