from owlready2 import *

# Importamos las librerías necesarias para trabajar con ontologías

# Cargamos o creamos una ontología
# En este caso, crearemos una nueva ontología para representar conceptos de astronomía
onto = get_ontology("http://example.org/astronomy_ontology.owl")

# Definimos las clases principales de la ontología
with onto:
    # Clase para representar objetos astronómicos
    class AstronomicalObject(Thing):
        pass

    # Subclases de AstronomicalObject
    class Star(AstronomicalObject):
        pass

    class Planet(AstronomicalObject):
        pass

    class Moon(AstronomicalObject):
        pass

    class Galaxy(AstronomicalObject):
        pass

    # Propiedades para relacionar objetos astronómicos
    class orbits(AstronomicalObject >> AstronomicalObject):
        pass

    class part_of(AstronomicalObject >> AstronomicalObject):
        pass

# Creamos instancias de los objetos astronómicos
with onto:
    sun = Star("Sun")
    earth = Planet("Earth")
    moon = Moon("Moon")
    milky_way = Galaxy("MilkyWay")

    # Definimos relaciones entre los objetos
    earth.orbits = [sun]
    moon.orbits = [earth]
    sun.part_of = [milky_way]
    earth.part_of = [milky_way]
    moon.part_of = [milky_way]

# Guardamos la ontología en un archivo OWL
onto.save(file="astronomy_ontology.owl", format="rdfxml")

# Consultamos la ontología
print("Objetos astronómicos en la ontología:")
for obj in onto.classes():
    print(obj)

print("\nRelaciones de órbita:")
for obj in onto.individuals():
    if hasattr(obj, "orbits"):
        print(f"{obj.name} orbita a {obj.orbits[0].name}")
