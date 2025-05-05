import cv2
import numpy as np

def detectar_movimiento_simple(frame1, frame2, umbral=20):
    """Detecta movimiento entre dos frames consecutivos."""
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, umbral, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def dibujar_contornos(frame, contornos):
    """Dibuja contornos alrededor del movimiento detectado."""
    for c in contornos:
        if cv2.contourArea(c) < 500:  # Ignorar movimiento pequeño
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # Abrir la cámara (o usar un archivo de video)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        exit()

    ret, frame1 = cap.read()
    if not ret:
        print("No se pudo leer el primer frame")
        exit()
    frame1 = cv2.resize(frame1, (640, 480))
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv2.resize(frame2, (640, 480))
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.GaussianBlur(frame2_gray, (21, 21), 0)

        contornos = detectar_movimiento_simple(frame1, frame2)
        frame_con_contornos = dibujar_contornos(frame2, contornos)

        cv2.imshow("Detección de Movimiento", frame_con_contornos)

        frame1 = frame2.copy()
        frame1_gray = frame2_gray.copy()

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
