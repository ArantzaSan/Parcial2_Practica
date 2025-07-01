# 133_Tipos_Razon_Aprendizaje.py

# Este programa implementa diferentes tipos de razonamiento en el aprendizaje:
# - Razonamiento inductivo
# - Razonamiento deductivo
# - Razonamiento abductivo

# Definimos una clase base para los tipos de razonamiento
class Razonamiento:
    def razonar(self, datos):
        raise NotImplementedError("Este método debe ser implementado por las subclases.")

# Razonamiento inductivo: Generaliza a partir de ejemplos específicos
class RazonamientoInductivo(Razonamiento):
    def razonar(self, ejemplos):
        # Supone que todos los ejemplos tienen una propiedad en común
        if not ejemplos:
            return "No hay ejemplos para inducir."
        propiedad = ejemplos[0]
        for ej in ejemplos:
            if ej != propiedad:
                return "No se puede inducir una regla general."
        return f"Todos los ejemplos tienen la propiedad: {propiedad}"

# Razonamiento deductivo: Aplica una regla general a un caso específico
class RazonamientoDeductivo(Razonamiento):
    def razonar(self, regla_general, caso):
        # Si el caso cumple la condición de la regla, deduce la conclusión
        condicion, conclusion = regla_general
        if condicion(caso):
            return f"Por deducción: {conclusion}"
        else:
            return "No se puede deducir la conclusión para este caso."

# Razonamiento abductivo: Busca la mejor explicación posible para una observación
class RazonamientoAbductivo(Razonamiento):
    def razonar(self, observacion, posibles_causas):
        # Retorna la primera causa que puede explicar la observación
        for causa, explicacion in posibles_causas:
            if causa(observacion):
                return f"La mejor explicación es: {explicacion}"
        return "No se encontró una explicación plausible."

# Ejemplo de uso de los tres tipos de razonamiento
if __name__ == "__main__":
    # Inductivo
    inductivo = RazonamientoInductivo()
    ejemplos = ["rojo", "rojo", "rojo"]
    print(inductivo.razonar(ejemplos))

    # Deductivo
    deductivo = RazonamientoDeductivo()
    regla = (lambda x: x > 18, "La persona es mayor de edad")
    caso = 20
    print(deductivo.razonar(regla, caso))

    # Abductivo
    abductivo = RazonamientoAbductivo()
    observacion = "tierra mojada"
    causas = [
        (lambda obs: obs == "tierra mojada", "Ha llovido"),
        (lambda obs: obs == "tierra seca", "No ha llovido")
    ]
    print(abductivo.razonar(observacion, causas))
