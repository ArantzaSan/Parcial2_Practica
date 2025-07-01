import clips

# Importamos la biblioteca de CLIPS

# Creamos un entorno de CLIPS
env = clips.Environment()

# Definimos las reglas y hechos relacionados con la lógica difusa para música
# Cargamos las definiciones en el entorno de CLIPS

# Definimos las categorías de música y sus características
env.build("""
(deftemplate Musica
    (slot genero)
    (slot energia)
    (slot estado_animo))
""")

# Definimos una regla para recomendar música basada en el estado de ánimo y nivel de energía
env.build("""
(defrule recomendar-musica
    (Musica (genero ?genero) (energia ?energia) (estado_animo ?estado_animo))
    (test (and (>= ?energia 7) (eq ?estado_animo "feliz")))
    =>
    (printout t "Recomendación: Escucha música de género " ?genero " con alta energía para mantener tu buen ánimo." crlf))
""")

# Insertamos hechos iniciales en el entorno
env.assert_string('(Musica (genero "Pop") (energia 8) (estado_animo "feliz"))')
env.assert_string('(Musica (genero "Jazz") (energia 4) (estado_animo "relajado"))')

# Ejecutamos el motor de inferencia para evaluar las reglas
env.run()

# Cerramos el entorno de CLIPS
env.clear()
