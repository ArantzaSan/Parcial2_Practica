
# 119_R_RedesSemanticas_LDescriptiva.py
# Implementación básica de una red semántica descriptiva en Python

# Definimos una clase Nodo para representar conceptos u objetos en la red
class Nodo:
    def __init__(self, nombre):
        self.nombre = nombre
        self.atributos = {}  # Diccionario de atributos y sus valores
        self.relaciones = {} # Diccionario de relaciones (tipo: lista de nodos)

    def agregar_atributo(self, atributo, valor):
        self.atributos[atributo] = valor

    def agregar_relacion(self, tipo, nodo_objetivo):
        if tipo not in self.relaciones:
            self.relaciones[tipo] = []
        self.relaciones[tipo].append(nodo_objetivo)

    def __str__(self):
        return self.nombre

# Función para mostrar la información de un nodo
def mostrar_nodo(nodo):
    print(f"Nodo: {nodo.nombre}")
    print("  Atributos:")
    for k, v in nodo.atributos.items():
        print(f"    {k}: {v}")
    print("  Relaciones:")
    for tipo, nodos in nodo.relaciones.items():
        print(f"    {tipo}: {[n.nombre for n in nodos]}")
    print()

# Reglas descriptivas: funciones que permiten inferir información
def es_un(nodo, tipo):
    # Regla: Si el nodo tiene una relación 'es_un' con el tipo dado
    return any(t.nombre == tipo for t in nodo.relaciones.get('es_un', []))

def tiene_atributo(nodo, atributo):
    # Regla: Si el nodo tiene el atributo dado
    return atributo in nodo.atributos

def obtener_valor_atributo(nodo, atributo):
    # Regla: Devuelve el valor del atributo si existe, sino busca en la jerarquía
    if atributo in nodo.atributos:
        return nodo.atributos[atributo]
    # Buscar en los nodos padres (herencia)
    for padre in nodo.relaciones.get('es_un', []):
        valor = obtener_valor_atributo(padre, atributo)
        if valor is not None:
            return valor
    return None

# Ejemplo de uso
if __name__ == "__main__":
    # Crear nodos
    animal = Nodo("Animal")
    animal.agregar_atributo("tiene_celulas", True)

    ave = Nodo("Ave")
    ave.agregar_relacion("es_un", animal)
    ave.agregar_atributo("tiene_plumas", True)

    canario = Nodo("Canario")
    canario.agregar_relacion("es_un", ave)
    canario.agregar_atributo("color", "amarillo")

    # Mostrar nodos
    mostrar_nodo(animal)
    mostrar_nodo(ave)
    mostrar_nodo(canario)

    # Inferencias usando reglas
    print("¿Canario es un Ave?", es_un(canario, "Ave"))
    print("¿Canario tiene plumas?", obtener_valor_atributo(canario, "tiene_plumas"))
    print("¿Canario tiene células?", obtener_valor_atributo(canario, "tiene_celulas"))
    print("¿Canario es un Animal?", es_un(canario, "Animal"))
