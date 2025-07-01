# Encadenamiento hacia adelante en lógica de primer orden aplicado a robótica

# Base de conocimiento: Reglas y hechos iniciales
# Aquí definimos las reglas y hechos que el sistema utilizará para razonar.
base_conocimiento = {
    "hechos": [
        "robot en posicion inicial",
        "bateria cargada",
        "sensor calibrado"
    ],
    "reglas": [
        {"si": ["robot en posicion inicial", "bateria cargada"], "entonces": "robot listo para moverse"},
        {"si": ["robot listo para moverse", "sensor calibrado"], "entonces": "robot puede explorar"},
        {"si": ["robot puede explorar"], "entonces": "robot inicia exploracion"}
    ]
}

# Función para verificar si todas las condiciones de una regla se cumplen
def condiciones_cumplidas(condiciones, hechos):
    return all(condicion in hechos for condicion in condiciones)

# Motor de inferencia: Encadenamiento hacia adelante
# Este motor evalúa las reglas y genera nuevos hechos a partir de las reglas aplicables.
def encadenamiento_hacia_adelante(base_conocimiento):
    hechos = set(base_conocimiento["hechos"])
    reglas = base_conocimiento["reglas"]
    nuevos_hechos = True

    while nuevos_hechos:
        nuevos_hechos = False
        for regla in reglas:
            if condiciones_cumplidas(regla["si"], hechos) and regla["entonces"] not in hechos:
                hechos.add(regla["entonces"])
                nuevos_hechos = True
                print(f"Nuevo hecho inferido: {regla['entonces']}")

    return hechos

# Programa principal
# Aquí se ejecuta el motor de inferencia y se muestran los resultados.
if __name__ == "__main__":
    print("Hechos iniciales:")
    for hecho in base_conocimiento["hechos"]:
        print(f"- {hecho}")

    print("\nInferencia en progreso...\n")
    hechos_finales = encadenamiento_hacia_adelante(base_conocimiento)

    print("\nHechos finales:")
    for hecho in hechos_finales:
        print(f"- {hecho}")
