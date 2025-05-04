import random

class SupervivenciaRobot:
    def __init__(self, energia, riesgo_ubicacion, recompensa_ubicacion):
        self.energia = energia
        self.riesgo_ubicacion = riesgo_ubicacion  # Probabilidad de hallar peligro en un nuevo lugar
        self.recompensa_ubicacion = recompensa_ubicacion  # Ganancia esperada en un nuevo lugar

    def calcular_utilidad_esperada(self, accion):
        if accion == "mover":
            # Utilidad esperada de trasladarse a un nuevo sitio
            utilidad_esperada = (self.recompensa_ubicacion * (1 - self.riesgo_ubicacion)) - (self.energia * self.riesgo_ubicacion)
        elif accion == "quedar":
            # Utilidad esperada de permanecer y recargar energia
            utilidad_esperada = self.energia * 0.6  # Supongamos que recargar da un 60% de la energia actual como utilidad
        return utilidad_esperada

    def valor_de_la_informacion(self):
        # Calcular la utilidad esperada de ambas acciones posibles
        utilidad_mover = self.calcular_utilidad_esperada("mover")
        utilidad_quedar = self.calcular_utilidad_esperada("quedar")

        # El valor de la información es la diferencia absoluta entre las utilidades esperadas
        voi = abs(utilidad_mover - utilidad_quedar)
        return voi

    def tomar_decision(self):
        voi = self.valor_de_la_informacion()
        print(f"Valor de la Información: {voi}")

        if voi > 12:  # Umbral arbitrario ajustado para la decisión de explorar
            print("El valor
