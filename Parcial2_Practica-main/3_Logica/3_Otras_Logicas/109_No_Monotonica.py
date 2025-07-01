from pyknow import *

# Importamos la biblioteca necesaria para trabajar con lógica no monótonica

# Definimos una clase que representa nuestro sistema experto
class ArteNoMonotonico(KnowledgeEngine):
    """
    Este sistema experto utiliza lógica no monótonica para recomendar estilos de arte
    basándose en las preferencias del usuario. La lógica no monótonica permite que
    las conclusiones cambien si se agregan nuevos hechos.
    """

    # Definimos una regla inicial para preguntar al usuario sobre sus preferencias
    @Rule(NOT(Fact(estilo=W())))
    def preguntar_estilo(self):
        self.declare(Fact(estilo=input("¿Qué estilo de arte prefieres? (abstracto, realista, surrealista): ").lower()))

    # Regla para recomendar arte abstracto
    @Rule(Fact(estilo='abstracto'))
    def recomendar_abstracto(self):
        print("Te recomendamos explorar obras de Kandinsky o Mondrian.")

    # Regla para recomendar arte realista
    @Rule(Fact(estilo='realista'))
    def recomendar_realista(self):
        print("Te recomendamos explorar obras de Da Vinci o Rembrandt.")

    # Regla para recomendar arte surrealista
    @Rule(Fact(estilo='surrealista'))
    def recomendar_surrealista(self):
        print("Te recomendamos explorar obras de Dalí o Magritte.")

    # Regla para manejar cambios en las preferencias del usuario
    @Rule(Fact(estilo=MATCH.estilo_anterior),
          NOT(Fact(estilo=MATCH.estilo_anterior)))
    def cambiar_preferencia(self, estilo_anterior):
        print(f"Has cambiado tu preferencia de {estilo_anterior}. Actualizando recomendación...")
        self.retract(Fact(estilo=estilo_anterior))
        self.declare(Fact(estilo=input("¿Qué nuevo estilo prefieres? (abstracto, realista, surrealista): ").lower()))

# Creamos una instancia del sistema experto
sistema = ArteNoMonotonico()

# Iniciamos el sistema experto
sistema.reset()  # Reseteamos el motor de inferencia
sistema.run()    # Ejecutamos las reglas definidas
