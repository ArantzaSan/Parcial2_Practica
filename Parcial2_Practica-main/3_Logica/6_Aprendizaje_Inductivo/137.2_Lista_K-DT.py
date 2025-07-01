# Lista K-DT orientada a música
# K-DT (Knowledge-Driven Table) es una estructura para almacenar y consultar conocimiento basado en reglas.

# Definimos una clase para representar una regla musical
class ReglaMusical:
    def __init__(self, condiciones, resultado):
        """
        condiciones: diccionario con atributos y valores (ej: {'genero': 'rock', 'instrumento': 'guitarra'})
        resultado: resultado asociado a la regla (ej: 'Recomendar solo de guitarra')
        """
        self.condiciones = condiciones
        self.resultado = resultado

    def coincide(self, consulta):
        """
        Verifica si la consulta coincide con las condiciones de la regla.
        """
        for clave, valor in self.condiciones.items():
            if clave not in consulta or consulta[clave] != valor:
                return False
        return True

# Clase para la lista K-DT
class ListaKDT:
    def __init__(self):
        self.reglas = []

    def agregar_regla(self, condiciones, resultado):
        """
        Agrega una nueva regla a la lista.
        """
        regla = ReglaMusical(condiciones, resultado)
        self.reglas.append(regla)

    def consultar(self, consulta):
        """
        Consulta la lista K-DT y retorna el resultado de la primera regla que coincida.
        """
        for regla in self.reglas:
            if regla.coincide(consulta):
                return regla.resultado
        return "No se encontró una recomendación para esa consulta."

# Ejemplo de uso
if __name__ == "__main__":
    # Crear la lista K-DT
    lista_kdt = ListaKDT()

    # Agregar reglas musicales
    lista_kdt.agregar_regla({'genero': 'rock', 'instrumento': 'guitarra'}, 'Recomendar solo de guitarra eléctrica')
    lista_kdt.agregar_regla({'genero': 'clasica', 'instrumento': 'piano'}, 'Recomendar concierto de piano')
    lista_kdt.agregar_regla({'genero': 'jazz', 'instrumento': 'saxofon'}, 'Recomendar improvisación de saxofón')

    # Consultar la lista K-DT
    consulta1 = {'genero': 'rock', 'instrumento': 'guitarra'}
    consulta2 = {'genero': 'clasica', 'instrumento': 'violin'}
    consulta3 = {'genero': 'jazz', 'instrumento': 'saxofon'}

    print(lista_kdt.consultar(consulta1))  # Recomendación para rock y guitarra
    print(lista_kdt.consultar(consulta2))  # No hay recomendación para clásica y violín
    print(lista_kdt.consultar(consulta3))  # Recomendación para jazz y saxofón
