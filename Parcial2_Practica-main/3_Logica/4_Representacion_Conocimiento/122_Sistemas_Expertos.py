# Sistema Experto Simple en Python
# Este programa implementa un sistema experto básico para diagnóstico de enfermedades.

# Base de conocimientos: reglas IF-THEN
reglas = [
    {
        "condiciones": {"fiebre": True, "tos": True, "dolor_cabeza": True},
        "diagnostico": "Gripe"
    },
    {
        "condiciones": {"fiebre": True, "erupcion": True},
        "diagnostico": "Sarampión"
    },
    {
        "condiciones": {"dolor_garganta": True, "tos": True},
        "diagnostico": "Resfriado"
    }
]

# Motor de inferencia: compara hechos con las reglas
def inferir(hechos):
    for regla in reglas:
        if all(hechos.get(cond, False) == valor for cond, valor in regla["condiciones"].items()):
            return regla["diagnostico"]
    return "No se pudo determinar el diagnóstico"

# Interfaz de usuario: solicita síntomas al usuario
def obtener_hechos():
    hechos = {}
    sintomas = ["fiebre", "tos", "dolor_cabeza", "erupcion", "dolor_garganta"]
    print("Responda 's' para sí y 'n' para no:")
    for sintoma in sintomas:
        respuesta = input(f"¿Tiene {sintoma.replace('_', ' ')}? (s/n): ").strip().lower()
        hechos[sintoma] = (respuesta == 's')
    return hechos

# Programa principal
if __name__ == "__main__":
    print("Sistema Experto de Diagnóstico Médico Simple")
    hechos = obtener_hechos()
    resultado = inferir(hechos)
    print(f"Diagnóstico sugerido: {resultado}")
