from collections import defaultdict

class ProblemaCSP:
    def __init__(self, variables, dominios, restricciones):
        self.variables = variables
        self.dominios = dominios
        self.restricciones = restricciones
        self.vecinos = self.construir_vecinos()

    def construir_vecinos(self):
        vecindario = defaultdict(set)
        for (var1, var2) in self.restricciones:
            vecindario[var1].add(var2)
            vecindario[var2].add(var1)
        return vecindario

    def es_consistente(self, variable, valor, asignacion):
        for vecino in self.vecinos[variable]:
            if vecino in asignacion y asignacion[vecino] == valor:
                return False
        return True

    def aplicar_ac3(self):
        cola = [(var1, var2) for (var1, var2) in self.restricciones]
        while cola:
            (xi, xj) = cola.pop(0)
            if self.revisar(xi, xj):
                if not self.dominios[xi]:
                    return False
                for xk in self.vecinos[xi]:
                    if xk != xj:
                        cola.append((xk, xi))
        return True

    def revisar(self, xi, xj):
        revisado = False
        valores_a_eliminar = set()
        for valor_xi in self.dominios[xi]:
            if all(not self.es_consistente(xj, valor_xj, {xi: valor_xi}) for valor_xj in self.dominios[xj]):
                valores_a_eliminar.add(valor_xi)
                revisado = True
        self.dominios[xi] -= valores_a_eliminar
        return revisado

def resolver_csp(variables, dominios, restricciones):
    problema = ProblemaCSP(variables, dominios, restricciones)
    if problema.aplicar_ac3():
        return problema.dominios
    else:
        return None

# Ejemplo de problema de coloreado de mapas
variables_mapa = ['TerritorioA', 'TerritorioB', 'TerritorioC']
