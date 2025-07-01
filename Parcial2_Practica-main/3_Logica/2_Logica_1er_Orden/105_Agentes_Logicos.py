from pyswip import Prolog

# Importamos las bibliotecas necesarias para trabajar con lógica de primer orden

# Inicializamos el motor de Prolog
prolog = Prolog()

# Base de conocimientos: Definimos hechos y reglas relacionados con la agricultura
# Hechos sobre cultivos y sus necesidades
prolog.assertz("cultivo(maiz)")
prolog.assertz("cultivo(trigo)")
prolog.assertz("necesita(maiz, agua)")
prolog.assertz("necesita(maiz, fertilizante)")
prolog.assertz("necesita(trigo, agua)")
prolog.assertz("necesita(trigo, fertilizante)")

# Hechos sobre las condiciones del suelo
prolog.assertz("suelo(rico_en_nutrientes)")
prolog.assertz("suelo(pobre_en_nutrientes)")

# Reglas para determinar si un cultivo puede crecer en un tipo de suelo
prolog.assertz("puede_crecer(Cultivo, rico_en_nutrientes) :- cultivo(Cultivo)")
prolog.assertz("puede_crecer(maiz, pobre_en_nutrientes) :- necesita(maiz, fertilizante)")
prolog.assertz("puede_crecer(trigo, pobre_en_nutrientes) :- necesita(trigo, fertilizante)")

# Reglas para determinar si un cultivo tiene todas sus necesidades cubiertas
prolog.assertz("necesidades_cubiertas(Cultivo) :- necesita(Cultivo, agua), necesita(Cultivo, fertilizante)")

# Consultas al sistema lógico
# Verificamos qué cultivos pueden crecer en suelos ricos en nutrientes
print("Cultivos que pueden crecer en suelos ricos en nutrientes:")
for resultado in prolog.query("puede_crecer(Cultivo, rico_en_nutrientes)"):
    print(f"- {resultado['Cultivo']}")

# Verificamos qué cultivos pueden crecer en suelos pobres en nutrientes
print("\nCultivos que pueden crecer en suelos pobres en nutrientes:")
for resultado in prolog.query("puede_crecer(Cultivo, pobre_en_nutrientes)"):
    print(f"- {resultado['Cultivo']}")

# Verificamos si las necesidades de un cultivo están cubiertas
print("\nCultivos con todas sus necesidades cubiertas:")
for resultado in prolog.query("necesidades_cubiertas(Cultivo)"):
    print(f"- {resultado['Cultivo']}")

# Fin del programa
# Este programa utiliza lógica de primer orden para modelar un sistema de agentes lógicos
# que ayuda a determinar las condiciones necesarias para el crecimiento de cultivos en la agricultura.
