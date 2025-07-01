from unification import unify, var

# Importamos la biblioteca `unification` para trabajar con lógica de primer orden

# Definimos una función principal para demostrar la inferencia por unificación
def main():
    # Definimos variables (representan incógnitas en lógica de primer orden)
    x = var('x')  # Variable genérica
    y = var('y')  # Otra variable genérica

    # Definimos hechos y reglas en el contexto de la física
    # Hecho 1: La fuerza es igual a la masa por la aceleración (F = m * a)
    fuerza = ('fuerza', 'masa', 'aceleracion')

    # Hecho 2: La energía cinética es igual a 1/2 * masa * velocidad^2 (E = 1/2 * m * v^2)
    energia_cinetica = ('energia_cinetica', 'masa', 'velocidad')

    # Regla: Si conocemos la masa y la aceleración, podemos calcular la fuerza
    regla_fuerza = ('fuerza', 10, 2)  # Ejemplo: masa = 10 kg, aceleración = 2 m/s^2

    # Regla: Si conocemos la masa y la velocidad, podemos calcular la energía cinética
    regla_energia = ('energia_cinetica', 5, 3)  # Ejemplo: masa = 5 kg, velocidad = 3 m/s

    # Intentamos unificar los hechos con las reglas
    unificacion_fuerza = unify(fuerza, regla_fuerza)
    unificacion_energia = unify(energia_cinetica, regla_energia)

    # Mostramos los resultados de la unificación
    print("Unificación de fuerza:", unificacion_fuerza)
    print("Unificación de energía cinética:", unificacion_energia)

# Ejecutamos la función principal
if __name__ == "__main__":
    main()
