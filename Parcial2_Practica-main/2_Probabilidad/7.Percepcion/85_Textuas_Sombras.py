import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters

def crear_textura_simple(tamaño=64, frecuencia=5):
    """Crea una textura sinusoidal simple."""
    x, y = np.meshgrid(np.linspace(0, 1, tamaño), np.linspace(0, 1, tamaño))
    textura = np.sin(2 * np.pi * frecuencia * x) * np.cos(2 * np.pi * frecuencia * y)
    return (textura + 1) / 2  # Normalizar a [0, 1]

def aplicar_sombra_simple(imagen, intensidad=0.5, direccion=(1, 1)):
    """Aplica una sombra direccional simple a una imagen."""
    h, w = imagen.shape[:2]
    sombra = np.zeros_like(imagen, dtype=float)
    dx, dy = direccion
    for i in range(h):
        for j in range(w):
            if (i * dy + j * dx) > (h * dy + w * dx) / 2:
                sombra[i, j] = -intensidad
    return np.clip(imagen + sombra, 0, 1)

if __name__ == "__main__":
    # Crear una textura de ejemplo
    textura_ejemplo = crear_textura_simple()

    # Aplicar una sombra a la textura
    textura_con_sombra = aplicar_sombra_simple(textura_ejemplo)

    # Cargar una imagen y aplicar una sombra
    imagen_ejemplo_gris = color.rgb2gray(data.camera())
    imagen_con_sombra = aplicar_sombra_simple(imagen_ejemplo_gris, intensidad=0.3, direccion=(-1, 1))

    # Mostrar los resultados
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(textura_ejemplo, cmap='gray')
    plt.title("Textura Simple")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(textura_con_sombra, cmap='gray')
    plt.title("Textura con Sombra")
    plt.axis('off')

    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(imagen_ejemplo_gris, cmap='gray')
    plt.title("Imagen Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagen_con_sombra, cmap='gray')
    plt.title("Imagen con Sombra")
    plt.axis('off')

    plt.show()
