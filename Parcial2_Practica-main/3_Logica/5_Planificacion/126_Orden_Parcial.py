import networkx as nx

# 126_Orden_Parcial.py
# Implementación de orden parcial como recurso de planificación
# Autor: GitHub Copilot

# Importamos la librería necesaria para trabajar con grafos dirigidos

# Definimos las tareas y sus dependencias (precondiciones)
# Cada tupla (A, B) indica que la tarea A debe realizarse antes que la tarea B
dependencias = [
    ('Cocinar', 'Comer'),
    ('Comprar ingredientes', 'Cocinar'),
    ('Poner la mesa', 'Comer'),
    ('Lavar platos', 'Guardar platos'),
    ('Comer', 'Lavar platos')
]

# Creamos un grafo dirigido para representar el orden parcial
orden_parcial = nx.DiGraph()
orden_parcial.add_edges_from(dependencias)

# Función para mostrar el orden parcial de las tareas
def mostrar_orden_parcial(grafo):
    print("Tareas y sus dependencias:")
    for pre, post in grafo.edges():
        print(f"- {pre} -> {post}")

# Función para obtener una secuencia válida de ejecución (orden topológico)
def obtener_planificacion(grafo):
    try:
        plan = list(nx.topological_sort(grafo))
        print("\nUna posible planificación válida:")
        for tarea in plan:
            print(f"- {tarea}")
    except nx.NetworkXUnfeasible:
        print("¡Error! Hay ciclos en las dependencias.")

# Función principal
def main():
    mostrar_orden_parcial(orden_parcial)
    obtener_planificacion(orden_parcial)

if __name__ == "__main__":
    main()
