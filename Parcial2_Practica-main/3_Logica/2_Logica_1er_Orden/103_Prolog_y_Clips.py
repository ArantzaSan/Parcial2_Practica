import clips

# Importamos la biblioteca de CLIPS

# Inicializamos el entorno de CLIPS
env = clips.Environment()

# Definimos las reglas y hechos en el entorno de CLIPS
# Estas reglas están orientadas a la toma de decisiones de un robot móvil

# Agregamos hechos iniciales al entorno
env.assert_string("(robot en-carga)")
env.assert_string("(bateria llena)")
env.assert_string("(obstaculo detectado)")

# Definimos las reglas en CLIPS
# Regla 1: Si el robot está en carga y la batería está llena, entonces puede moverse
env.build("""
(defrule mover-robot
    (robot en-carga)
    (bateria llena)
    =>
    (assert (robot puede-moverse))
    (printout t "El robot puede moverse." crlf))
""")

# Regla 2: Si el robot detecta un obstáculo, entonces debe detenerse
env.build("""
(defrule detener-robot
    (obstaculo detectado)
    =>
    (assert (robot detenerse))
    (printout t "El robot debe detenerse debido a un obstáculo." crlf))
""")

# Regla 3: Si el robot no detecta obstáculos y puede moverse, entonces avanza
env.build("""
(defrule avanzar-robot
    (not (obstaculo detectado))
    (robot puede-moverse)
    =>
    (assert (robot avanzar))
    (printout t "El robot está avanzando." crlf))
""")

# Ejecutamos las reglas en el entorno
env.run()

# Mostramos los hechos finales en el entorno
print("Hechos finales:")
for fact in env.facts():
    print(fact)
