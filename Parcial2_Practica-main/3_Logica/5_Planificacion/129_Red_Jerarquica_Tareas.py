# 129_Red_Jerarquica_Tareas.py
# Programa para implementar una red jerárquica de tareas como recurso de planificación

# Definimos la clase Tarea para representar cada tarea en la jerarquía
class Tarea:
    def __init__(self, nombre, descripcion, subtareas=None):
        self.nombre = nombre
        self.descripcion = descripcion
        self.subtareas = subtareas if subtareas else []

    # Método para agregar una subtarea
    def agregar_subtarea(self, subtarea):
        self.subtareas.append(subtarea)

    # Método para mostrar la jerarquía de tareas de forma recursiva
    def mostrar(self, nivel=0):
        print('  ' * nivel + f"- {self.nombre}: {self.descripcion}")
        for subtarea in self.subtareas:
            subtarea.mostrar(nivel + 1)

# Función principal para crear y mostrar la red jerárquica de tareas
def main():
    # Creamos tareas principales
    tarea_principal = Tarea("Proyecto IA", "Desarrollo de un sistema de IA")

    # Creamos subtareas de primer nivel
    analisis = Tarea("Análisis", "Recolección y análisis de requisitos")
    diseno = Tarea("Diseño", "Diseño de la arquitectura del sistema")
    implementacion = Tarea("Implementación", "Codificación y pruebas")

    # Agregamos subtareas a la tarea principal
    tarea_principal.agregar_subtarea(analisis)
    tarea_principal.agregar_subtarea(diseno)
    tarea_principal.agregar_subtarea(implementacion)

    # Creamos subtareas de segundo nivel
    analisis.agregar_subtarea(Tarea("Entrevistas", "Entrevistas con usuarios"))
    analisis.agregar_subtarea(Tarea("Documentación", "Elaboración de documentos de requisitos"))

    diseno.agregar_subtarea(Tarea("Modelo de datos", "Diseño de la base de datos"))
    diseno.agregar_subtarea(Tarea("Arquitectura", "Definición de módulos y componentes"))

    implementacion.agregar_subtarea(Tarea("Frontend", "Desarrollo de la interfaz de usuario"))
    implementacion.agregar_subtarea(Tarea("Backend", "Desarrollo de la lógica de negocio"))

    # Mostramos la red jerárquica de tareas
    print("Red jerárquica de tareas:")
    tarea_principal.mostrar()

if __name__ == "__main__":
    main()        
