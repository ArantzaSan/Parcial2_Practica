import cv2
import numpy as np

def reconocer_objeto_simple(imagen_gris, plantilla, umbral=0.8):
    """Realiza un reconocimiento de objeto simple usando coincidencia de plantillas."""
    resultado = cv2.matchTemplate(imagen_gris, plantilla, cv2.TM_CCOEFF_NORMED)
    loc = np.where(resultado >= umbral)
    coordenadas = list(zip(*loc[::-1]))
    return coordenadas

if __name__ == '__main__':
    # Crear una imagen de ejemplo (cuadrado blanco sobre fondo negro)
    imagen = np.zeros((100, 100), dtype=np.uint8)
    imagen[20:40, 20:40] = 255

    # Crear una plantilla del objeto a buscar (un cuadrado más pequeño)
    plantilla = np.ones((10, 10), dtype=np.uint8) * 255

    # Introducir múltiples ocurrencias del objeto en una imagen más grande
    imagen_grande = np.zeros((150, 150), dtype=np.uint8)
    imagen_grande[10:30, 10:30] = plantilla
    imagen_grande[80:100, 120:140] = plantilla

    # Realizar el reconocimiento de objetos
    ubicaciones = reconocer_objeto_simple(imagen_grande, plantilla)

    # Dibujar rectángulos alrededor de los objetos detectados
    imagen_color = cv2.cvtColor(imagen_grande, cv2.COLOR_GRAY2BGR)
    w, h = plantilla.shape[::-1]
    for pt in ubicaciones:
        cv2.rectangle(imagen_color, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow('Objetos Detectados', imagen_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
