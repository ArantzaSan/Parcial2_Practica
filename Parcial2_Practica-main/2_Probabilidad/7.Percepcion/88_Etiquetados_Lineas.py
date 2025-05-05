import cv2
import numpy as np

def etiquetar_lineas_hough(imagen_gris, min_longitud=50, max_gap=10):
    """Detecta líneas usando la Transformada de Hough y las etiqueta."""
    bordes = cv2.Canny(imagen_gris, 50, 150)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, 100, minLineLength=min_longitud, maxLineGap=max_gap)
    if lineas is not None:
        imagen_color = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
        for i, linea in enumerate(lineas):
            x1, y1, x2, y2 = linea[0]
            color = (0, 255, 0)  # Verde
            cv2.line(imagen_color, (x1, y1), (x2, y2), color, 2)
            # Etiquetar la línea con su índice
            texto = str(i + 1)
            org = ((x1 + x2) // 2, (y1 + y2) // 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 0, 0)  # Rojo
            thickness = 1
            cv2.putText(imagen_color, texto, org, font, font_scale, font_color, thickness, cv2.LINE_AA)
        return imagen_color
    return cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

if __name__ == '__main__':
    # Crear una imagen de ejemplo con algunas líneas
    imagen = np.zeros((200, 300), dtype=np.uint8)
    cv2.line(imagen, (50, 50), (250, 50), 255, 5)
    cv2.line(imagen, (100, 20), (100, 180), 255, 5)
    cv2.line(imagen, (20, 150), (280, 180), 255, 5)

    # Etiquetar las líneas detectadas
    imagen_etiquetada = etiquetar_lineas_hough(imagen)

    # Mostrar la imagen original y la etiquetada
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Líneas Etiquetadas', imagen_etiquetada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
