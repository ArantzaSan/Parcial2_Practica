from sympy import symbols, And, Or, Not, Implies, Function, satisfiable

# Resolución de Skolem en lógica de primer orden aplicada al campo de los videojuegos

# Importamos las bibliotecas necesarias

# Definimos las funciones y predicados que utilizaremos
# Por ejemplo, "EsJugador(x)" indica si x es un jugador
EsJugador = Function('EsJugador')
Juega = Function('Juega')  # "Juega(x, y)" indica si el jugador x juega al videojuego y
EsCompetitivo = Function('EsCompetitivo')  # "EsCompetitivo(y)" indica si el videojuego y es competitivo

# Definimos las constantes y variables
x, y, z = symbols('x y z')  # Variables genéricas
Videojuego1, Videojuego2 = symbols('Videojuego1 Videojuego2')  # Constantes para videojuegos

# Definimos las premisas en lógica de primer orden
# Premisa 1: Todos los jugadores juegan al menos un videojuego
premisa1 = Implies(EsJugador(x), Or(Juega(x, Videojuego1), Juega(x, Videojuego2)))

# Premisa 2: Si un jugador juega un videojuego competitivo, entonces es competitivo
premisa2 = Implies(And(EsJugador(x), Juega(x, y)), EsCompetitivo(y))

# Premisa 3: Videojuego1 es competitivo
premisa3 = EsCompetitivo(Videojuego1)

# Negamos la conclusión que queremos probar
# Conclusión: Existe un jugador que juega Videojuego1 y es competitivo
conclusion = And(Juega(x, Videojuego1), EsCompetitivo(Videojuego1))
negacion_conclusion = Not(conclusion)

# Combinamos las premisas con la negación de la conclusión
# Esto se hace para aplicar la resolución y verificar si el conjunto es insatisfacible
conjunto = And(premisa1, premisa2, premisa3, negacion_conclusion)

# Verificamos la satisfacibilidad del conjunto
resultado = satisfiable(conjunto)

# Mostramos el resultado
if resultado:
    print("El conjunto es satisfacible. La conclusión no se puede probar.")
    print("Ejemplo de modelo que satisface el conjunto:", resultado)
else:
    print("El conjunto es insatisfacible. La conclusión es válida.")
