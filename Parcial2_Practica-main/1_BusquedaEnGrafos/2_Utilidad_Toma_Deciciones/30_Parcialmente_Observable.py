import random

class AstronomiaPOMDP:
    def __init__(self, estados, acciones, observaciones, probabilidades_transicion, probabilidades_observacion, recompensas, factor_descuento=0.9):
        self.estados = estados  # Estados posibles (condiciones del cielo)
        self.acciones = acciones  # Acciones posibles (observar estrella o galaxia)
        self.observaciones = observaciones  # Observaciones posibles (clima aparente)
        self.probabilidades_transicion = probabilidades_transicion  # Probabilidades de transición
        self.probabilidades_observacion = probabilidades_observacion  # Probabilidades de observación
        self.recompensas = recompensas  # Recompensas inmediatas por acción en cada estado
        self.factor_descuento = factor_descuento  # Factor de descuento
        self.estado_creencia = {estado: 1/len(estados) for estado in estados}  # Estado de creencia inicial uniforme

    def actualizar_estado_creencia(self, accion, observacion):
        # Actualizar el estado de creencia basado en la acción y la observación
        nuevo_creencia = {}
        for siguiente_estado in self.estados:
            probabilidad = 0
            for estado in self.estados:
                prob_transicion = self.probabilidades_transicion[estado][accion][siguiente_estado]
                prob_observacion = self.probabilidades_observacion[siguiente_estado][observacion]
                probabilidad += self.estado_creencia[estado] * prob_transicion * prob_observacion
            nuevo_creencia[siguiente_estado] = probabilidad

        # Normalizar el estado de creencia
        total = sum(nuevo_creencia.values())
        self.estado_creencia = {estado: prob / total for estado, prob in nuevo_creencia.items()}

    def elegir_accion(self):
        # Elegir la mejor acción basada en el estado de creencia actual
        valores_acciones = {}
        for accion in self.acciones:
            valor = sum(
                self.estado_creencia[estado] * (self.recompensas[estado][accion] + self.factor_descuento * max(
                    sum(self.probabilidades_transicion[estado][accion][siguiente_estado] * self.estado_creencia[siguiente_estado]
                        for siguiente_estado in self.estados),
                    0
                ))
                for estado in self.estados
            )
            valores_acciones[accion] = valor
        return max(valores_acciones, key=valores_acciones.get)

    def simular(self, pasos=5):
        # Simular la toma de decisiones durante un número de pasos
        for _ in range(pasos):
            accion = self.elegir_accion()
            observacion = random.choice(self.observaciones)  # Simular una observación
            print(f"Acción elegida: {accion}, Observación: {observacion}")
            self.actualizar_estado_creencia(accion, observacion)
            print(f"Estado de creencia actual: {self.estado_creencia}")

# Definir estados, acciones, observaciones, recompensas y probabilidades
estados_cielo = ['claro', 'nublado', 'lluvioso']  # Condiciones del cielo
acciones_observacion = ['observar_estrella', 'observar_galaxia']  # Acciones posibles
observaciones_clima = ['cielo_claro', 'cielo_nublado', 'cielo_lluvioso']  # Observaciones posibles

# Recompensas inmediatas por acción en cada estado
recompensas_observacion_astronomica = {
    'claro': {'observar_estrella': 10, 'observar_galaxia': 8},
    'nublado': {'observar_estrella': 5, 'observar_galaxia': 4},
    'lluvioso': {'observar_estrella': 1, 'observar_galaxia': 2}
}

# Probabilidades de transición de estado dada una acción
probabilidades_transicion_clima = {
    'claro': {
        'observar_estrella': {'claro': 0.7, 'nublado': 0.2, 'lluvioso': 0.1},
        'observar_galaxia': {'claro': 0.6, 'nublado': 0.3, 'lluvioso': 0.1}
    },
    'nublado': {
        'observar_estrella': {'claro': 0.3, 'nublado': 0.5, 'lluvioso': 0.2},
        'observar_galaxia': {'claro': 0.2, 'nublado': 0.6, 'lluvioso': 0.2}
    },
    'lluvioso': {
        'observar_estrella': {'claro': 0.1, 'nublado': 0.3, 'lluvioso': 0.6},
        'observar_galaxia': {'claro': 0.1, 'nublado': 0.2, 'lluvioso': 0.7}
    }
}

# Probabilidades de observación dado un estado
probabilidades_observacion_clima = {
    'claro': {'cielo_claro': 0.8, 'cielo_nublado': 0.15, 'cielo_lluvioso': 0.05},
    'nublado': {'cielo_claro': 0.2, 'cielo_nublado': 0.7, 'cielo_lluvioso': 0.1},
    'lluvioso': {'cielo_claro': 0.05, 'cielo_nublado': 0.25, 'cielo_lluvioso': 0.7}
}

# Crear el POMDP de astronomía y simular la toma de decisiones
pomdp_astronomia = AstronomiaPOMDP(estados_cielo, acciones_observacion, observaciones_clima, probabilidades_transicion_clima, probabilidades_observacion_clima, recompensas_observacion_astronomica)
pomdp_astronomia.simular(pasos=5)
