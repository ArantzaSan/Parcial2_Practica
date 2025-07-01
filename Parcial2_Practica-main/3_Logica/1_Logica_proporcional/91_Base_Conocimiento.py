from pyknow import *

# Importamos la librería necesaria para trabajar con lógica proposicional

# Definimos una clase que representa nuestra Base de Conocimiento
class LogicaProposicional(KnowledgeEngine):
    """
    Esta clase hereda de KnowledgeEngine de la librería pyknow.
    Aquí definimos las reglas y hechos que forman nuestra base de conocimiento.
    """

    # Definimos una regla básica
    @Rule(Fact(p=True), Fact(q=True))
    def regla_implicacion(self):
        """
        Si 'p' es verdadero y 'q' es verdadero, entonces imprimimos que 'p implica q'.
        """
        print("La proposición 'p implica q' es verdadera.")

    # Otra regla para manejar negaciones
    @Rule(Fact(p=False))
    def regla_negacion(self):
        """
        Si 'p' es falso, entonces imprimimos que 'p' es falso.
        """
        print("La proposición 'p' es falsa.")

    # Una regla adicional para conjunciones
    @Rule(Fact(p=True), Fact(q=True))
    def regla_conjuncion(self):
        """
        Si 'p' y 'q' son verdaderos, entonces imprimimos que 'p y q' es verdadero.
        """
        print("La proposición 'p y q' es verdadera.")

# Función principal para ejecutar el programa
if __name__ == "__main__":
    # Creamos una instancia de nuestra base de conocimiento
    motor = LogicaProposicional()

    # Reseteamos el motor para inicializarlo
    motor.reset()

    # Declaramos hechos iniciales
    motor.declare(Fact(p=True))
    motor.declare(Fact(q=True))

    # Ejecutamos el motor para evaluar las reglas
    motor.run()
