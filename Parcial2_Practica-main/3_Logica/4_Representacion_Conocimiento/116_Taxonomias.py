# Taxonomías como representación del conocimiento en paleontología
# Este programa utiliza una estructura jerárquica para representar taxonomías de dinosaurios.
# Cada nivel de la jerarquía representa una categoría taxonómica (Reino, Filo, Clase, Orden, Familia, Género, Especie).

# Definimos una clase para representar nodos en la taxonomía
class Taxon:
    def __init__(self, nombre, descripcion):
        """
        Inicializa un nodo de la taxonomía.
        :param nombre: Nombre del taxón (ej. "Dinosauria").
        :param descripcion: Breve descripción del taxón.
        """
        self.nombre = nombre
        self.descripcion = descripcion
        self.subtaxones = []  # Lista de subtaxones (hijos)

    def agregar_subtaxon(self, subtaxon):
        """
        Agrega un subtaxón (hijo) al taxón actual.
        :param subtaxon: Instancia de Taxon que representa el subtaxón.
        """
        self.subtaxones.append(subtaxon)

    def mostrar_taxonomia(self, nivel=0):
        """
        Muestra la taxonomía de forma jerárquica.
        :param nivel: Nivel de profundidad en la jerarquía (usado para indentación).
        """
        print("  " * nivel + f"{self.nombre}: {self.descripcion}")
        for subtaxon in self.subtaxones:
            subtaxon.mostrar_taxonomia(nivel + 1)

# Construcción de la taxonomía de dinosaurios
def construir_taxonomia_dinosaurios():
    """
    Construye una taxonomía básica de dinosaurios.
    :return: Nodo raíz de la taxonomía.
    """
    # Nodo raíz: Reino
    reino_animalia = Taxon("Animalia", "Reino que incluye a todos los animales.")

    # Filo
    filo_chordata = Taxon("Chordata", "Filo que incluye animales con notocorda.")
    reino_animalia.agregar_subtaxon(filo_chordata)

    # Clase
    clase_reptilia = Taxon("Reptilia", "Clase que incluye reptiles, incluidos los dinosaurios.")
    filo_chordata.agregar_subtaxon(clase_reptilia)

    # Orden
    orden_dinosauria = Taxon("Dinosauria", "Orden que incluye a los dinosaurios.")
    clase_reptilia.agregar_subtaxon(orden_dinosauria)

    # Familias y géneros de ejemplo
    familia_tyrannosauridae = Taxon("Tyrannosauridae", "Familia de dinosaurios terópodos grandes.")
    orden_dinosauria.agregar_subtaxon(familia_tyrannosauridae)

    genero_tyrannosaurus = Taxon("Tyrannosaurus", "Género que incluye al famoso T. rex.")
    familia_tyrannosauridae.agregar_subtaxon(genero_tyrannosaurus)

    especie_tyrannosaurus_rex = Taxon("Tyrannosaurus rex", "Uno de los depredadores más grandes de la historia.")
    genero_tyrannosaurus.agregar_subtaxon(especie_tyrannosaurus_rex)

    return reino_animalia

# Programa principal
if __name__ == "__main__":
    # Construimos la taxonomía
    taxonomia_dinosaurios = construir_taxonomia_dinosaurios()

    # Mostramos la taxonomía completa
    print("Taxonomía de Dinosaurios:")
    taxonomia_dinosaurios.mostrar_taxonomia()
