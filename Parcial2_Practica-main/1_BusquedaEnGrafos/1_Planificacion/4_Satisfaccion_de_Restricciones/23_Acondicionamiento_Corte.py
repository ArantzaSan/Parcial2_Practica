from itertools import product

def resolver_acondicionamiento_cutset(variables, dominios, restricciones, cutset):
    # Generar todas las combinaciones posibles para las variables del cutset
    asignaciones_cutset = list(product(*[dominios[var] for var in cutset]))

    for asignacion in asignaciones_cutset:
        solucion_cutset = dict(zip(cutset, asignacion))

        # Verificar si la asignación del cutset satisface todas las restricciones
        if all(restriccion(solucion_cutset) for restriccion in restricciones):
            # Intentar resolver el resto del problema con el cutset ya asignado
            solucion_restante = busqueda_con_retroceso(variables, dominios, restricciones, solucion_cutset)
            if solucion_restante:
                return {**solucion_cutset, **solucion_restante}

    return None

def busqueda_con_retroceso(variables, dominios, restricciones, asignacion_actual):
    if len(asignacion_actual) == len(variables):
        return asignacion_actual

    variables_sin_asignar = [var for var in variables if var not in asignacion_actual]
    variable_actual = variables_sin_asignar[0]

    for valor in dominios[variable_actual]:
        nueva_asignacion = {**asignacion_actual, variable_actual: valor}
        if all(restriccion(nueva_asignacion) for restriccion in restricciones):
            resultado = busqueda_con_retroceso(variables, dominios, restricciones, nueva_asignacion)
            if resultado:
                return resultado

    return None

# Definir variables, dominios y restricciones para la planificación de tareas
tareas = ['TareaA', 'TareaB', 'TareaC']
duraciones = {
    'TareaA': [1, 2],
    'TareaB': [2, 3],
    'TareaC': [1, 2]
}

# Restricción: la suma de las duraciones de las tareas debe ser menor o igual a 5
def restriccion_duracion_total(asignacion):
    if len(asignacion) == len(tareas):
        return sum(asignacion.values()) <= 5
    return True

condiciones = [restriccion_duracion_total]

# Definir el cutset (una tarea para condicionar)
cutset_tareas = ['TareaA']

# Resolver el problema de planificación de tareas utilizando acondicionamiento de corte
solucion_planificacion = resolver_acondicionamiento_cutset(tareas, duraciones, condiciones, cutset_tareas)

if solucion_planificacion:
    print(f"Solución de planificación encontrada: {solucion_planificacion}")
else:
    print("No se encontró solución de planificación.")
