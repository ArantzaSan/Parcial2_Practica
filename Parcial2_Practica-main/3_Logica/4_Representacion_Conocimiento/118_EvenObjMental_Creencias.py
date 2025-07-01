from datetime import datetime

# Importamos las bibliotecas necesarias
# `datetime` se usará para manejar marcas de tiempo en los eventos

# Clase Evento
# Representa un evento que ocurre en un momento específico
class Evento:
    def __init__(self, nombre, descripcion):
        self.nombre = nombre  # Nombre del evento
        self.descripcion = descripcion  # Descripción del evento
        self.timestamp = datetime.now()  # Marca de tiempo del evento

    def __str__(self):
        return f"Evento: {self.nombre}, Descripción: {self.descripcion}, Ocurrido en: {self.timestamp}"

# Clase ObjetoMental
# Representa un objeto mental, como una creencia
class ObjetoMental:
    def __init__(self, tipo, contenido):
        self.tipo = tipo  # Tipo del objeto mental (e.g., "creencia", "deseo")
        self.contenido = contenido  # Contenido del objeto mental

    def __str__(self):
        return f"ObjetoMental: Tipo: {self.tipo}, Contenido: {self.contenido}"

# Clase SistemaDeCreencias
# Administra un conjunto de creencias y registra eventos relacionados
class SistemaDeCreencias:
    def __init__(self):
        self.creencias = []  # Lista de objetos mentales de tipo "creencia"
        self.eventos = []  # Lista de eventos registrados

    # Método para agregar una nueva creencia
    def agregar_creencia(self, contenido):
        nueva_creencia = ObjetoMental(tipo="creencia", contenido=contenido)
        self.creencias.append(nueva_creencia)
        print(f"Creencia agregada: {nueva_creencia}")

    # Método para registrar un evento
    def registrar_evento(self, nombre, descripcion):
        nuevo_evento = Evento(nombre, descripcion)
        self.eventos.append(nuevo_evento)
        print(f"Evento registrado: {nuevo_evento}")

    # Método para mostrar todas las creencias
    def mostrar_creencias(self):
        print("Creencias actuales:")
        for creencia in self.creencias:
            print(creencia)

    # Método para mostrar todos los eventos
    def mostrar_eventos(self):
        print("Eventos registrados:")
        for evento in self.eventos:
            print(evento)

# Ejemplo de uso del sistema
if __name__ == "__main__":
    sistema = SistemaDeCreencias()

    # Agregamos creencias
    sistema.agregar_creencia("El cielo es azul")
    sistema.agregar_creencia("El agua hierve a 100 grados Celsius")

    # Registramos eventos
    sistema.registrar_evento("Inicio del sistema", "El sistema de creencias ha iniciado")
    sistema.registrar_evento("Nueva creencia", "Se agregó una nueva creencia sobre el cielo")

    # Mostramos las creencias y eventos
    sistema.mostrar_creencias()
    sistema.mostrar_eventos()
