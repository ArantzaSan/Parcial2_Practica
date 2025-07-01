# 121_Incertidumbre_FacCerteza.py
# Programa para representar incertidumbre usando factores de certeza en el contexto de geología

# -------------------------------
# Sección 1: Definición de reglas y hechos
# -------------------------------

# Definimos una base de conocimientos simple sobre tipos de rocas y sus características
reglas = [
    {
        "nombre": "Roca sedimentaria",
        "condiciones": {"granulometria": "fina", "color": "claro", "dureza": "baja"},
        "fc": 0.8  # Factor de certeza de la regla
    },
    {
        "nombre": "Roca ígnea",
        "condiciones": {"granulometria": "gruesa", "color": "oscuro", "dureza": "alta"},
        "fc": 0.9
    },
    {
        "nombre": "Roca metamórfica",
        "condiciones": {"granulometria": "media", "color": "variable", "dureza": "media"},
        "fc": 0.7
    }
]

# -------------------------------
# Sección 2: Función para calcular el factor de certeza
# -------------------------------

def calcular_fc_regla(regla, hechos):
    """
    Calcula el factor de certeza de una regla dada la evidencia (hechos).
    """
    coincidencias = 0
    total = len(regla["condiciones"])
    for clave, valor in regla["condiciones"].items():
        if hechos.get(clave) == valor:
            coincidencias += 1
    # El factor de certeza se ajusta según el porcentaje de coincidencias
    fc_evidencia = coincidencias / total
    fc_final = regla["fc"] * fc_evidencia
    return fc_final

# -------------------------------
# Sección 3: Entrada de datos del usuario
# -------------------------------

print("Sistema experto geológico: Identificación de tipo de roca con incertidumbre\n")
granulometria = input("Ingrese la granulometría (fina/media/gruesa): ").strip().lower()
color = input("Ingrese el color (claro/oscuro/variable): ").strip().lower()
dureza = input("Ingrese la dureza (baja/media/alta): ").strip().lower()

hechos = {
    "granulometria": granulometria,
    "color": color,
    "dureza": dureza
}

# -------------------------------
# Sección 4: Evaluación de reglas y presentación de resultados
# -------------------------------

resultados = []
for regla in reglas:
    fc = calcular_fc_regla(regla, hechos)
    if fc > 0:
        resultados.append((regla["nombre"], fc))

# Ordenamos los resultados por el factor de certeza descendente
resultados.sort(key=lambda x: x[1], reverse=True)

print("\nResultados de identificación con factores de certeza:")
if resultados:
    for nombre, fc in resultados:
        print(f"- {nombre}: FC = {fc:.2f}")
else:
    print("No se pudo identificar el tipo de roca con la información proporcionada.")

# -------------------------------
# Fin del programa
# -------------------------------
