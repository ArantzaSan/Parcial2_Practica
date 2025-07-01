# STRIPS (Stanford Research Institute Problem Solver) para planificación en una empresa
# Este programa implementa un planificador STRIPS simple para organizar tareas en una empresa.

# Definición de una Acción STRIPS
class Accion:
    def __init__(self, nombre, precondiciones, agregar, eliminar):
        self.nombre = nombre
        self.precondiciones = set(precondiciones)
        self.agregar = set(agregar)
        self.eliminar = set(eliminar)

    def es_aplicable(self, estado):
        # Verifica si la acción puede aplicarse en el estado actual
        return self.precondiciones.issubset(estado)

    def aplicar(self, estado):
        # Aplica la acción al estado y retorna el nuevo estado
        nuevo_estado = estado.copy()
        nuevo_estado -= self.eliminar
        nuevo_estado |= self.agregar
        return nuevo_estado

# Planificador STRIPS
def strips_planificador(estado_inicial, objetivo, acciones):
    plan = []
    estado = estado_inicial.copy()
    while not objetivo.issubset(estado):
        for accion in acciones:
            if accion.es_aplicable(estado) and not accion.agregar.issubset(estado):
                plan.append(accion.nombre)
                estado = accion.aplicar(estado)
                break
        else:
            # No se encontró acción aplicable, no se puede alcanzar el objetivo
            return None
    return plan

# Ejemplo de uso: Organización de una empresa
# Estado inicial: solo hay una tarea pendiente y nadie asignado
estado_inicial = {
    "tarea_pendiente",
    "empleado_disponible"
}

# Objetivo: tarea completada
objetivo = {
    "tarea_completada"
}

# Definición de acciones posibles
acciones = [
    Accion(
        "asignar_tarea",
        precondiciones=["tarea_pendiente", "empleado_disponible"],
        agregar=["tarea_asignada"],
        eliminar=["empleado_disponible"]
    ),
    Accion(
        "realizar_tarea",
        precondiciones=["tarea_asignada"],
        agregar=["tarea_completada"],
        eliminar=["tarea_pendiente", "tarea_asignada"]
    ),
    Accion(
        "liberar_empleado",
        precondiciones=["tarea_completada"],
        agregar=["empleado_disponible"],
        eliminar=[]
    )
]

# Ejecución del planificador
plan = strips_planificador(estado_inicial, objetivo, acciones)

# Mostrar el plan generado
if plan:
    print("Plan generado para alcanzar el objetivo:")
    for paso in plan:
        print("-", paso)
else:
    print("No se pudo encontrar un plan para alcanzar el objetivo.")

# Comentarios:
# - La clase Accion modela las acciones STRIPS con precondiciones, efectos de adición y eliminación.
# - strips_planificador busca un plan secuencial para alcanzar el objetivo desde el estado inicial.
# - El ejemplo simula la asignación y realización de una tarea en una empresa.
