# 117_AccSitEvent_Marcos.py
# Este programa utiliza acciones, situaciones, eventos y marcos como representación de conocimiento.
# Se desarrolla una historia simple para ilustrar cómo estas representaciones pueden ser utilizadas.

# Definimos un marco para representar un lugar (una habitación)
class Room:
    def __init__(self, name, description, objects=None):
        self.name = name
        self.description = description
        self.objects = objects if objects else []

    def __str__(self):
        return f"{self.name}: {self.description}. Objetos: {', '.join(self.objects)}"

# Definimos un marco para representar un personaje
class Character:
    def __init__(self, name, location):
        self.name = name
        self.location = location

    def move_to(self, new_location):
        print(f"{self.name} se mueve de {self.location.name} a {new_location.name}.")
        self.location = new_location

# Definimos un marco para representar un evento
class Event:
    def __init__(self, description, action):
        self.description = description
        self.action = action

    def trigger(self):
        print(f"Evento: {self.description}")
        self.action()

# Creamos las habitaciones (situaciones iniciales)
kitchen = Room("Cocina", "Una cocina limpia con una mesa y una silla", ["manzana", "cuchillo"])
living_room = Room("Sala de estar", "Una sala acogedora con un sofá y una televisión", ["libro", "control remoto"])

# Creamos un personaje
alice = Character("Alicia", kitchen)

# Definimos una acción como parte de un evento
def alice_finds_apple():
    if "manzana" in alice.location.objects:
        print(f"{alice.name} encuentra una manzana en la {alice.location.name}.")
        alice.location.objects.remove("manzana")
    else:
        print(f"No hay manzana en la {alice.location.name}.")

# Creamos un evento que desencadena la acción
event_find_apple = Event("Alicia busca una manzana", alice_finds_apple)

# Historia principal
def main():
    # Situación inicial
    print("Situación inicial:")
    print(kitchen)
    print(living_room)
    print()

    # Alicia busca una manzana en la cocina
    event_find_apple.trigger()
    print()

    # Alicia se mueve a la sala de estar
    alice.move_to(living_room)
    print()

    # Nueva situación
    print("Situación final:")
    print(kitchen)
    print(living_room)

if __name__ == "__main__":
    main()
