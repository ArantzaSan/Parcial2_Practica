import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters, segmentation, measure
from skimage.segmentation import slic
from skimage.color import label2rgb

def detectar_aristas(imagen):
    """Detecta bordes en una imagen usando el filtro de Sobel."""
    bordes = filters.sobel(imagen)
    return bordes

def segmentar_slic(imagen, n_segmentos=100, compactness=10):
    """Segmenta una imagen usando el algoritmo SLIC."""
    segmentos = slic(imagen, n_segments=n_segmentos, compactness=compactness, start_label=1)
    return segmentos

def mostrar_segmentacion(imagen_original, segmentos):
    """Muestra la imagen original con los límites de los segmentos superpuestos."""
    bordes = segmentation.mark_boundaries(imagen_original, segmentos)
    plt.imshow(bordes)
    plt.title("Segmentación SLIC")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Cargar una imagen de ejemplo (RGB)
    imagen_color = data.astronaut()
    imagen_gris = color.rgb2gray(imagen_color)

    # Detección de aristas
    bordes_detectados = detectar_aristas(imagen_gris)

    # Segmentación SLIC
    segmentos_slic = segmentar_slic(imagen_color)

    # Mostrar resultados
    plt.imshow(bordes_detectados, cmap='gray')
    plt.title("Detección de Aristas (Sobel)")
    plt.axis('off')
    plt.show()

    mostrar_segmentacion(imagen_color, segmentos_slic)
