class OptimizacionInventario:
    def __init__(self, inventario_inicial, inventario_maximo, costo_almacenamiento, costo_compra, precio_venta, probabilidades_demanda):
        self.inventario_inicial = inventario_inicial
        self.inventario_maximo = inventario_maximo
        self.costo_almacenamiento = costo_almacenamiento
        self.costo_compra = costo_compra
        self.precio_venta = precio_venta
        self.probabilidades_demanda = probabilidades_demanda  # Probabilidades de demanda diaria
        self.funcion_valor = [0] * (inventario_maximo + 1)  # Función de valor inicializada a 0

    def recompensa_esperada(self, estado, accion):
        # Calcular la recompensa esperada de tomar una acción en un estado dado
        recompensa = 0
        nuevo_inventario = estado + accion

        # Costo de compra y almacenamiento
        recompensa -= accion * self.costo_compra
        recompensa -= nuevo_inventario * self.costo_almacenamiento

        # Calcular la recompensa esperada para cada nivel de demanda
        for demanda, probabilidad in self.probabilidades_demanda.items():
            ventas = min(nuevo_inventario, demanda)
            recompensa += probabilidad * (ventas * self.precio_venta)
            nuevo_inventario -= ventas

        return recompensa, nuevo_inventario

    def iteracion_valor(self, factor_descuento=0.9, tolerancia=1e-3):
        # Iteración de valores
        while True:
            delta = 0
            for estado in range(self.inventario_maximo + 1):
                valor_maximo = float('-inf')
                for accion in range(self.inventario_maximo - estado + 1):
                    recompensa, nuevo_inventario = self.recompensa_esperada(estado, accion)
                    valor = recompensa + factor_descuento * self.funcion_valor[nuevo_inventario]
                    if valor > valor_maximo:
                        valor_maximo = valor
                delta = max(delta, abs(self.funcion_valor[estado] - valor_maximo))
                self.funcion_valor[estado] = valor_maximo
            if delta < tolerancia:
                break

    def politica_optima(self):
        politica = {}
        for estado in range(self.inventario_maximo + 1):
            valor_maximo = float('-inf')
            mejor_accion = 0
            for accion in range(self.inventario_maximo - estado + 1):
                recompensa, nuevo_inventario = self.recompensa_esperada(estado, accion)
                valor = recompensa + self.funcion_valor[nuevo_inventario]
                if valor > valor_maximo:
                    valor_maximo = valor
                    mejor_accion = accion
            politica[estado] = mejor_accion
        return politica

# Parámetros del inventario de la tienda
inventario_inicial = 5
inventario_maximo = 10
costo_mantener = 1  # Costo de mantener un artículo en inventario por día
costo_adquisicion = 3  # Costo de comprar un artículo
precio_venta_articulo = 5  # Precio de venta de un artículo
probabilidad_venta_diaria = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}  # Probabilidades de demanda diaria

# Crear el optimizador de inventario y ejecutar la iteración de valores
optimizador = OptimizacionInventario(inventario_inicial, inventario_maximo, costo_mantener, costo_adquisicion, precio_venta_articulo, probabilidad_venta_diaria)
optimizador.iteracion_valor()
politica_optima = optimizador.politica_optima()

# Mostrar la política óptima
print("Política óptima de compra de inventario:")
for estado, accion in politica_optima.items():
    print(f"Si el inventario actual es {estado}, la cantidad óptima a comprar es {accion}.")
