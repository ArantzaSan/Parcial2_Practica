import numpy as np
import random

class ControlVoltajeMDP:
    def __init__(self, estados, acciones, probabilidades_transicion, recompensas, factor_descuento=0.9):
        self.estados = estados  # Estados posibles (niveles de voltaje)
        self.acciones = acciones  # Acciones posibles (ajustes de voltaje)
        self.probabilidades_transicion = probabilidades_transicion  # Probabilidades de transición
        self.recompensas = recompensas  # Recompensas inmediatas por acción en cada estado
        self.factor_descuento = factor_descuento  # Factor de descuento
        self.funcion_valor = {estado: 0 for estado in estados}  # Función de valor inicial
        self.politica = {estado: random.choice(acciones) for estado in estados}  # Política inicial aleatoria

    def evaluar_politica(self):
        # Evaluar la política actual
        while True:
            variacion = 0
            for estado in self.estados:
                valor_anterior = self.funcion_valor[estado]
                accion = self.politica[estado]
                valor_actualizado = self.recompensas[estado][accion] + self.factor_descuento * sum(
                    self.probabilidades_transicion[estado][accion][estado_siguiente] * self.funcion_valor[estado_siguiente]
                    for estado_siguiente in self.estados
                )
                self.funcion_valor[estado] = valor_actualizado
                variacion = max(variacion, abs(valor_anterior - valor_actualizado))
            if variacion < 1e-3:
                break

    def mejorar_politica(self):
        # Mejorar la política basada en la función de valor actual
        politica_estable = True
        for estado in self.estados:
            accion_previa = self.politica[estado]
            valores_acciones = {}
            for accion in self.acciones:
                valores_acciones[accion] = self.recompensas[estado][accion] + self.factor_descuento * sum(
                    self.probabilidades_transicion[estado][accion][estado_siguiente] * self.funcion_valor[estado_siguiente]
                    for estado_siguiente in self.estados
                )
            mejor_accion = max(valores_acciones, key=valores_acciones.get)
            self.politica[estado] = mejor_accion
            if accion_previa != self.politica[estado]:
                politica_estable = False
        return politica_estable

    def iteracion_politica(self):
        # Iteración de políticas
        while True:
            self.evaluar_politica()
            if self.mejorar_politica():
                break

# Definir estados, acciones, recompensas y probabilidades de transición
estados_voltaje = ['bajo', 'normal', 'alto']  # Niveles de voltaje
ajustes_voltaje = ['disminuir', 'mantener', 'aumentar']  # Ajustes de voltaje

# Recompensas inmediatas por acción en cada estado
recompensas_inmediatas_voltaje = {
    'bajo': {'disminuir': -10, 'mantener': 0, 'aumentar': 5},
    'normal': {'disminuir': 0, 'mantener': 5, 'aumentar': 0},
    'alto': {'disminuir': 5, 'mantener': 0, 'aumentar': -10}
}

# Probabilidades de transición de estado dada una acción
probabilidades_transicion_voltaje = {
    'bajo': {
        'disminuir': {'bajo': 0.9, 'normal': 0.1, 'alto': 0.0},
        'mantener': {'bajo': 0.6, 'normal': 0.4, 'alto': 0.0},
        'aumentar': {'bajo': 0.2, 'normal': 0.7, 'alto': 0.1}
    },
    'normal': {
        'disminuir': {'bajo': 0.3, 'normal': 0.6, 'alto': 0.1},
        'mantener': {'bajo': 0.1, 'normal': 0.8, 'alto': 0.1},
        'aumentar': {'bajo': 0.1, 'normal': 0.4, 'alto': 0.5}
    },
    'alto': {
        'disminuir': {'bajo': 0.1, 'normal': 0.7, 'alto': 0.2},
        'mantener': {'bajo': 0.0, 'normal': 0.4, 'alto': 0.6},
        'aumentar': {'bajo': 0.0, 'normal': 0.1, 'alto': 0.9}
    }
}

# Crear el controlador de voltaje y ejecutar la iteración de políticas
controlador_voltaje = ControlVoltajeMDP(estados_voltaje, ajustes_voltaje, probabilidades_transicion_voltaje, recompensas_inmediatas_voltaje)
controlador_voltaje.iteracion_politica()

# Mostrar la política óptima
print("Política óptima de control de voltaje:")
for estado, accion in controlador_voltaje.politica.items():
    print(f"Si el voltaje es {estado}, la acción óptima es {accion}.")
