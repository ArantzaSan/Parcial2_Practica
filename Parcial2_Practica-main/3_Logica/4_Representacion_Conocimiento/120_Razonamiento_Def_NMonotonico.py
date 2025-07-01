# 120_Razonamiento_Def_NMonotonico.py
# Ejemplo de razonamiento monotónico en el dominio de la medicina
# El razonamiento monotónico implica que, al agregar nueva información, las conclusiones previas no se invalidan.

# Definimos una base de conocimientos simple sobre síntomas y diagnósticos

class BaseConocimiento:
    def __init__(self):
        # Hechos conocidos (síntomas observados)
        self.hechos = set()
        # Reglas de inferencia (si ... entonces ...)
        self.reglas = []

    def agregar_hecho(self, hecho):
        self.hechos.add(hecho)

    def agregar_regla(self, condiciones, conclusion):
        self.reglas.append((set(condiciones), conclusion))

    def inferir(self):
        # Inferencia monotónica: las conclusiones solo se agregan, nunca se eliminan
        nuevos = True
        conclusiones = set()
        while nuevos:
            nuevos = False
            for condiciones, conclusion in self.reglas:
                if condiciones.issubset(self.hechos) and conclusion not in self.hechos:
                    self.hechos.add(conclusion)
                    conclusiones.add(conclusion)
                    nuevos = True
        return conclusiones

# Ejemplo de uso en medicina

# 1. Crear la base de conocimiento
bc = BaseConocimiento()

# 2. Agregar hechos observados (síntomas del paciente)
bc.agregar_hecho("fiebre")
bc.agregar_hecho("tos")

# 3. Agregar reglas médicas (simplificadas)
# Si fiebre y tos, entonces posible gripe
bc.agregar_regla(["fiebre", "tos"], "posible_gripe")
# Si fiebre y dolor_garganta, entonces posible faringitis
bc.agregar_regla(["fiebre", "dolor_garganta"], "posible_faringitis")
# Si posible_gripe, entonces recomendar reposo
bc.agregar_regla(["posible_gripe"], "recomendar_reposo")

# 4. Inferir diagnósticos y recomendaciones
conclusiones = bc.inferir()

# 5. Mostrar resultados
print("Hechos observados y conclusiones inferidas:")
for hecho in bc.hechos:
    print("-", hecho)

# 6. Si agregamos un nuevo síntoma, las conclusiones anteriores se mantienen (razonamiento monotónico)
print("\nAgregando nuevo síntoma: dolor_garganta")
bc.agregar_hecho("dolor_garganta")
nuevas_conclusiones = bc.inferir()

print("Hechos y conclusiones actualizadas:")
for hecho in bc.hechos:
    print("-", hecho)
