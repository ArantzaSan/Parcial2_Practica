# 130_Condicional.py
# Ejemplo de uso de condicionales como recurso de planificación

# Definimos una función que planifica una actividad según el clima y la hora
def planificar_actividad(clima, hora):
    # Si está soleado y es temprano, salimos a correr
    if clima == "soleado" and hora < 10:
        print("Plan: Salir a correr por el parque.")
    # Si está nublado y es mediodía, vamos al gimnasio
    elif clima == "nublado" and 10 <= hora < 14:
        print("Plan: Ir al gimnasio.")
    # Si está lloviendo, quedarse en casa y leer un libro
    elif clima == "lluvioso":
        print("Plan: Quedarse en casa y leer un libro.")
    # Si es tarde, descansar
    elif hora >= 20:
        print("Plan: Descansar y prepararse para dormir.")
    # Otras condiciones
    else:
        print("Plan: Realizar tareas pendientes o relajarse.")

# Sección principal del programa
if __name__ == "__main__":
    # Solicitamos al usuario el clima actual
    clima = input("¿Cómo está el clima? (soleado/nublado/lluvioso): ").strip().lower()
    # Solicitamos la hora actual
    try:
        hora = int(input("¿Qué hora es? (0-23): "))
    except ValueError:
        print("Hora inválida. Usa un número entre 0 y 23.")
        exit(1)
    # Llamamos a la función de planificación
    planificar_actividad(clima, hora)
