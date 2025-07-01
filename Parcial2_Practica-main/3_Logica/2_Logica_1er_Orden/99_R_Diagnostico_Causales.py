from pyke import knowledge_engine, krb_traceback

# Importamos la biblioteca necesaria para trabajar con lógica de primer orden

# Inicializamos el motor de inferencia
engine = knowledge_engine.engine(__file__)

# Cargamos las reglas y hechos en el motor
def cargar_reglas_y_hechos():
    """
    Esta función carga las reglas y hechos necesarios para el diagnóstico de fallas en sistemas electrónicos.
    """
    try:
        engine.reset()  # Reiniciamos el motor para cargar nuevas reglas y hechos
        engine.activate('diagnostico')  # Activamos el conjunto de reglas llamado 'diagnostico'
    except Exception as e:
        print("Error al cargar reglas y hechos:", e)

# Definimos una función para realizar el diagnóstico
def diagnosticar_falla():
    """
    Esta función utiliza las reglas de lógica de primer orden para diagnosticar fallas en un sistema electrónico.
    """
    try:
        # Solicitamos al motor que realice una inferencia basada en las reglas y hechos cargados
        with engine.prove_goal('diagnostico.identificar_falla($falla)') as resultado:
            for r in resultado:
                print(f"Falla identificada: {r['falla']}")
    except Exception as e:
        krb_traceback.print_exc()
        print("Error durante el diagnóstico:", e)

# Programa principal
if __name__ == "__main__":
    # Cargamos las reglas y hechos
    cargar_reglas_y_hechos()

    # Ejecutamos el diagnóstico
    print("Iniciando diagnóstico de fallas en el sistema electrónico...")
    diagnosticar_falla()
