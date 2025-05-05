import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Ejemplo Simple de Red Neuronal para Clasificación ---

def crear_modelo_clasificacion(input_shape, num_clases):
    """Crea un modelo secuencial simple para clasificación."""
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_clases, activation="softmax"), # Softmax para distribución de probabilidad
        ]
    )
    return modelo

def entrenar_modelo_clasificacion(modelo, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """Entrena un modelo de clasificación con optimizador y función de pérdida probabilística."""
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    historial = modelo.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    return historial

def evaluar_modelo_clasificacion(modelo, x_test, y_test):
    """Evalúa el modelo de clasificación."""
    perdida, precision = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"Pérdida en el conjunto de prueba: {perdida:.4f}")
    print(f"Precisión en el conjunto de prueba: {precision:.4f}")

# --- Ejemplo Simple de Red Neuronal para Regresión ---

def crear_modelo_regresion(input_shape):
    """Crea un modelo secuencial simple para regresión."""
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(1), # Salida lineal para regresión
        ]
    )
    return modelo

def entrenar_modelo_regresion(modelo, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """Entrena un modelo de regresión con optimizador y función de pérdida probabilística (MSE)."""
    modelo.compile(optimizer="adam", loss="mse") # MSE se relaciona con la varianza del error
    historial = modelo.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    return historial

def evaluar_modelo_regresion(modelo, x_test, y_test):
    """Evalúa el modelo de regresión."""
    perdida = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"Pérdida cuadrática media en el conjunto de prueba: {perdida:.4f}")

# --- Aprendizaje Probabilístico en Redes Profundas ---

# 1. Salidas Probabilísticas:
#    - Capa Softmax para clasificación: La capa final con activación softmax produce una distribución de probabilidad sobre las clases.
#    - Distribuciones de probabilidad para regresión: Se pueden modelar las salidas de regresión como parámetros de una distribución (ej., media y varianza de una Gaussiana) para tener incertidumbre.

def crear_modelo_regresion_probabilistico(input_shape):
    """Modelo de regresión que predice la media y la varianza."""
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(2), # Salida de 2 unidades: [media, log_varianza]
        ]
    )
    return modelo

def perdida_negativa_log_verosimilitud_gaussiana(y_verdadero, parametros_predichos):
    """Función de pérdida para regresión probabilística con distribución Gaussiana."""
    media = parametros_predichos[:, 0]
    log_varianza = parametros_predichos[:, 1]
    varianza = tf.exp(log_varianza)
    return tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * varianza) + 0.5 * tf.math.square(y_verdadero - media) / varianza)

def muestrear_de_gaussiana(parametros):
    """Muestrea de una distribución Gaussiana dada la media y la log-varianza."""
    media = parametros[:, 0]
    log_varianza = parametros[:, 1]
    varianza = tf.exp(log_varianza)
    epsilon = tf.random.normal(shape=tf.shape(media))
    return media + tf.sqrt(varianza) * epsilon

def entrenar_modelo_regresion_probabilistico(modelo, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """Entrena un modelo de regresión probabilístico."""
    modelo.compile(optimizer="adam", loss=perdida_negativa_log_verosimilitud_gaussiana)
    historial = modelo.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    return historial

# --- Ejemplo de Uso con Datos Sintéticos ---

if __name__ == "__main__":
    # --- Datos de Clasificación ---
    num_clases_clasificacion = 3
    num_muestras = 100
    dim_entrada = 2
    X_clasificacion, y_clasificacion = make_blobs(n_samples=num_muestras, centers=num_clases_clasificacion, n_features=dim_entrada, random_state=42)
    y_clasificacion = to_categorical(y_clasificacion, num_classes=num_clases_clasificacion)
    x_train_clasif, x_test_clasif, y_train_clasif, y_test_clasif = train_test_split(X_clasificacion, y_clasificacion, test_size=0.2, random_state=42)
    x_train_clasif, x_val_clasif, y_train_clasif, y_val_clasif = train_test_split(x_train_clasif, y_train_clasif, test_size=0.25, random_state=42)

    modelo_clasif = crear_modelo_clasificacion(input_shape=(dim_entrada,), num_clases=num_clases_clasificacion)
    historial_clasif = entrenar_modelo_clasificacion(modelo_clasif, x_train_clasif, np.argmax(y_train_clasif, axis=1),
                                                    x_val_clasif, np.argmax(y_val_clasif, axis=1), epochs=20)
    evaluar_modelo_clasificacion(modelo_clasif, x_test_clasif, np.argmax(y_test_clasif, axis=1))

    # --- Datos de Regresión ---
    num_muestras_regresion = 100
    X_regresion = np.linspace(-3, 3, num_muestras_regresion).reshape(-1, 1)
    y_regresion = np.sin(X_regresion) + np.random.normal(0, 0.5, num_muestras_regresion).reshape(-1, 1)
    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(X_regresion, y_regresion, test_size=0.2, random_state=42)
    x_train_reg, x_val_reg, y_train_reg, y_val_reg = train_test_split(x_train_reg, y_train_reg, test_size=0.25, random_state=42)

    modelo_reg = crear_modelo_regresion(input_shape=(1,))
    historial_reg = entrenar_modelo_regresion(modelo_reg, x_train_reg, y_train_reg, x_val_reg, y_val_reg, epochs=20)
    evaluar_modelo_regresion(modelo_reg, x_test_reg, y_test_reg)

    # --- Datos de Regresión Probabilística ---
    modelo_reg_prob = crear_modelo_regresion_probabilistico(input_shape=(1,))
    historial_reg_prob = entrenar_modelo_regresion_probabilistico(modelo_reg_prob, x_train_reg, y_train_reg, x_val_reg, y_val_reg, epochs=50)

    # Hacer predicciones probabilísticas
    parametros_predichos = modelo_reg_prob.predict(x_test_reg)
    medias_predichas = parametros_predichos[:, 0]
    log_varianzas_predichas = parametros_predichos[:, 1]
    varianzas_predichas = np.exp(log_varianzas_predichas)
    desviaciones_predichas = np.sqrt(varianzas_predichas)

    print("\nPredicciones de Regresión Probabilística (Media y Desviación Estándar):")
    for i in range(5):
        print(f"Entrada: {x_test_reg[i][0]:.2f}, Media Predicha: {medias_predichas[i]:.2f}, Desviación Estándar Predicha: {desviaciones_predichas[i]:.2f}, Real: {y_test_reg[i][0]:.2f}")

    # Visualizar las predicciones con incertidumbre
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test_reg, y_test_reg, color='blue', label='Datos Reales')
    plt.plot(x_test_reg, medias_predichas, color='red', label='Media Predicha')
    plt.fill_between(x_test_reg.flatten(), medias_predichas - 1.96 * desviaciones_predichas,
                     medias_predichas + 1.96 * desviaciones_predichas, color='red', alpha=0.2, label='Intervalo de Confianza (95%)')
    plt.xlabel('Entrada')
    plt.ylabel('Salida')
    plt.title('Regresión Probabilística con Red Neuronal')
    plt.legend()
    plt.grid(True)
    plt.show()
