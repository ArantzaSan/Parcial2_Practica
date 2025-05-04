class GestionEmpresa:
    def __init__(self, estados, acciones, recompensas, probabilidades_transicion, factor_descuento=0.9):
        self.estados = estados  # Estados posibles (niveles de ingresos)
        self.acciones = acciones  # Acciones posibles (niveles de inversión en marketing)
        self.recompensas = recompensas  # Recompensas inmediatas por acción en cada estado
        self.probabilidades_transicion = probabilidades_transicion  # Probabilidades de transición
        self.factor_descuento = factor_descuento  # Factor de descuento
        self.politica = {estado: acciones[0] for estado in estados}  # Política inicial
        self.funcion_valor = {estado: 0 for estado in estados}  # Función de valor inicial

    def evaluar_politica(self):
        # Evaluar la política actual
        while True:
            delta = 0
            for estado in self.estados:
                valor_antiguo = self.funcion_valor[estado]
                accion = self.politica[estado]
                nuevo_valor = self.recompensas[estado][accion] + self.factor_descuento * sum(
                    self.probabilidades_transicion[estado][accion][estado_siguiente] * self.funcion_valor[estado_siguiente]
                    for estado_siguiente in self.estados
                )
                self.funcion_valor[estado] = nuevo_valor
                delta = max(delta, abs(valor_antiguo - nuevo_valor))
            if delta < 1e-3:
                break

    def mejorar_politica(self):
        # Mejorar la política basada en la función de valor actual
        politica_estable = True
        for estado in self.estados:
            accion_antigua = self.politica[estado]
            valores_accion = {}
            for accion in self.acciones:
                valores_accion[accion] = self.recompensas[estado][accion] + self.factor_descuento * sum(
                    self.probabilidades_transicion[estado][accion][estado_siguiente] * self.funcion_valor[estado_siguiente]
                    for estado_siguiente in self.estados
                )
            mejor_accion = max(valores_accion, key=valores_accion.get)
            self.politica[estado] = mejor_accion
            if accion_antigua != self.politica[estado]:
                politica_estable = False
        return politica_estable

    def iteracion_politica(self):
        # Iteración de políticas
        while True:
            self.evaluar_politica()
            if self.mejorar_politica():
                break

# Definir estados, acciones, recompensas y probabilidades de transición
estados_empresa = ['bajo', 'medio', 'alto']  # Niveles de ingresos
acciones_marketing = ['invertir_poco', 'invertir_medio', 'invertir_mucho']  # Niveles de inversión en marketing

# Recompensas inmediatas por acción en cada estado
recompensas_empresa = {
    'bajo': {'invertir_poco': 10, 'invertir_medio': 20, 'invertir_mucho': 25},
    'medio': {'invertir_poco': 30, 'invertir_medio': 40, 'invertir_mucho': 45},
    'alto': {'invertir_poco': 50, 'invertir_medio': 55, 'invertir_mucho': 60}
}

# Probabilidades de transición de estado dada una acción
probabilidades_transicion_empresa = {
    'bajo': {
        'invertir_poco': {'bajo': 0.7, 'medio': 0.2, 'alto': 0.1},
        'invertir_medio': {'bajo': 0.5, 'medio': 0.3, 'alto': 0.2},
        'invertir_mucho': {'bajo': 0.4, 'medio': 0.3, 'alto': 0.3}
    },
    'medio': {
        'invertir_poco': {'bajo': 0.3, 'medio': 0.4, 'alto': 0.3},
        'invertir_medio': {'bajo': 0.2, 'medio': 0.5, 'alto': 0.3},
        'invertir_mucho': {'bajo': 0.2, 'medio': 0.3, 'alto': 0.5}
    },
    'alto': {
        'invertir_poco': {'bajo': 0.2, 'medio': 0.3, 'alto': 0.5},
        'invertir_medio': {'bajo': 0.1, 'medio': 0.3, 'alto': 0.6},
        'invertir_mucho': {'bajo': 0.1, 'medio': 0.2, 'alto': 0.7}
    }
}

# Crear el administrador de la empresa y ejecutar la iteración de políticas
administrador_empresa = GestionEmpresa(estados_empresa, acciones_marketing, recompensas_empresa, probabilidades_transicion_empresa)
administrador_empresa.iteracion_politica()

# Mostrar la política óptima
print("Política óptima de inversión en marketing para la empresa:")
for estado, accion in administrador_empresa.politica.items():
    print(f"Si los ingresos son {estado}, la inversión óptima en marketing es {accion}.")
