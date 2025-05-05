import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage import data, color

def aplicar_filtro(imagen, filtro):
    """Aplica un filtro 2D a una imagen."""
    imagen_filtrada = convolve2d(imagen, filtro, mode='same', boundary='fill', fillvalue=0)
    return imagen_filtrada

def mostrar_imagen(imagen, titulo="Imagen"):
    """Muestra una imagen usando matplotlib."""
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Cargar una imagen de ejemplo (en escala de grises)
    imagen_original = color.rgb2gray(data.camera())

    # Definir algunos filtros comunes
    filtro_promedio = np.ones((3, 3)) / 9
    filtro_gaussiano = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16
    filtro_realce_bordes = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]])

    # Aplicar los filtros
    imagen_promediada = aplicar_filtro(imagen_original, filtro_promedio)
    imagen_gaussiana = aplicar_filtro(imagen_original, filtro_gaussiano)
    imagen_bordes = aplicar_filtro(imagen_original, filtro_realce_bordes)

    # Mostrar las imágenes
    mostrar_imagen(imagen_original, "Imagen Original")
    mostrar_imagen(imagen_promediada, "Filtro Promedio")
    mostrar_imagen(imagen_gaussiana, "Filtro Gaussiano")
    mostrar_imagen(imagen_bordes, "Filtro Realce de Bordes")
