from collections import defaultdict

def restringir_factor(factor, variable, valor):
    """Restringe un factor a un valor específico de una variable."""
    factor_restringido = {}
    for asignacion, prob in factor.items():
        if asignacion.get(variable) == valor:
            nueva_asignacion = {k: v for k, v in asignacion.items() if k != variable}
            factor_restringido[tuple(sorted(nueva_asignacion.items()))] = prob
    return factor_restringido

def multiplicar_factores(factor1, factor2):
    """Multiplica dos factores."""
    resultado = defaultdict(float)
    for asignacion1, prob1 in factor1.items():
        dict_asignacion1 = dict(asignacion1)
        for asignacion2, prob2 in factor2.items():
            dict_asignacion2 = dict(asignacion2)
            variables_comunes = set(dict_asignacion1) & set(dict_asignacion2)
            if all(dict_asignacion1.get(var) == dict_asignacion2.get(var) for var in variables_comunes):
                nueva_asignacion_dict = {**dict_asignacion1, **dict_asignacion2}
                resultado[tuple(sorted(nueva_asignacion_dict.items()))] += prob1 * prob2
    return resultado

def sumar_variable(factor, variable):
    """Suma una variable de un factor."""
    factor_sumado = defaultdict(float)
    for asignacion, prob in factor.items():
        nueva_asignacion_dict = {k: v for k, v in dict(asignacion).items() if k != variable}
        factor_sumado[tuple(sorted(nueva_asignacion_dict.items()))] += prob
    return factor_sumado

def eliminacion_de_variables(factores, variables_consulta, variables_ocultas):
    """Realiza la eliminación de variables."""
    factores_procesados = list(factores)
    for var in variables_ocultas:
        # Multiplicar todos los factores que contienen la variable
        factores_relevantes = [factor for factor in factores_procesados if any(var in dict(asignacion) for asignacion in factor)]
        if not factores_relevantes:
            continue
        producto = factores_relevantes[0]
        for factor in factores_relevantes[1:]:
            producto = multiplicar_factores(producto, factor)
        # Sumar la variable
        sumado = sumar_variable(producto, var)
        # Eliminar los factores antiguos y añadir el nuevo
        factores_procesados = [factor for factor in factores_procesados if factor not in factores_relevantes]
        factores_procesados.append(sumado)
    # Multiplicar los factores restantes
    resultado = factores_procesados[0]
    for factor in factores_procesados[1:]:
        resultado = multiplicar_factores(resultado, factor)
    # Normalizar el resultado
    probabilidad_total = sum(resultado.values())
    resultado_normalizado = {k: v / probabilidad_total for k, v in resultado.items()}
    return resultado_normalizado

# Ejemplo de uso
if __name__ == "__main__":
    # Definir factores como diccionarios
    factor_a = {('A', True): 0.2, ('A', False): 0.8}
    factor_ab = {('A', True, 'B', True): 0.5, ('A', True, 'B', False): 0.5,
                 ('A', False, 'B', True): 0.3, ('A', False, 'B', False): 0.7}
    factores_ejemplo = [factor_a, factor_ab]
    variables_consulta_ejemplo = ['B']
    variables_ocultas_ejemplo = ['A']

    resultado_eliminacion = eliminacion_de_variables(factores_ejemplo, variables_consulta_ejemplo, variables_ocultas_ejemplo)
    print("Resultado:", resultado_eliminacion)
