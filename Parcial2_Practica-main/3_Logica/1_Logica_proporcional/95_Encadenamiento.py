# Encadenamiento hacia adelante en lógica proposicional

# Definimos una base de conocimiento con reglas y hechos iniciales
base_conocimiento = {
    "hechos": {"A", "B"},  # Hechos iniciales
    "reglas": [  # Reglas en forma de implicaciones
        {"condiciones": {"A", "B"}, "conclusion": "C"},
        {"condiciones": {"C"}, "conclusion": "D"},
        {"condiciones": {"D", "E"}, "conclusion": "F"}
    ]
}

# Función para verificar si una regla es aplicable
def es_regla_aplicable(regla, hechos):
    """
    Verifica si todas las condiciones de una regla están en los hechos conocidos.
    """
    return regla["condiciones"].issubset(hechos)

# Función para realizar el encadenamiento hacia adelante
def encadenamiento_hacia_adelante(base_conocimiento):
    """
    Realiza el encadenamiento hacia adelante para derivar nuevos hechos.
    """
    hechos = base_conocimiento["hechos"].copy()  # Copiamos los hechos iniciales
    reglas = base_conocimiento["reglas"]  # Obtenemos las reglas
    nuevos_hechos = True  # Bandera para controlar el proceso

    while nuevos_hechos:
        nuevos_hechos = False  # Reiniciamos la bandera
        for regla in reglas:
            if es_regla_aplicable(regla, hechos) and regla["conclusion"] not in hechos:
                # Si la regla es aplicable y la conclusión no está en los hechos
                hechos.add(regla["conclusion"])  # Añadimos la conclusión a los hechos
                nuevos_hechos = True  # Indicamos que se derivaron nuevos hechos
                print(f"Nuevo hecho derivado: {regla['conclusion']}")  # Mostramos el nuevo hecho

    return hechos

# Ejecutamos el encadenamiento hacia adelante
hechos_derivados = encadenamiento_hacia_adelante(base_conocimiento)

# Mostramos los hechos finales
print("Hechos finales derivados:", hechos_derivados)
