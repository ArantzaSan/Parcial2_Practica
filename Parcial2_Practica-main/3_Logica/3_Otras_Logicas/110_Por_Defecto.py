import argparse

# Importamos el módulo argparse para manejar argumentos de línea de comandos

# Definimos una función principal que ejecutará la lógica del programa
def main():
    # Configuramos el analizador de argumentos
    parser = argparse.ArgumentParser(description="Programa que utiliza lógica por defecto.")
    
    # Agregamos un argumento opcional con un valor por defecto
    parser.add_argument(
        '--nombre', 
        type=str, 
        default='Usuario', 
        help='Nombre del usuario (por defecto: Usuario)'
    )
    
    # Agregamos otro argumento opcional con un valor por defecto
    parser.add_argument(
        '--edad', 
        type=int, 
        default=18, 
        help='Edad del usuario (por defecto: 18)'
    )
    
    # Parseamos los argumentos proporcionados por el usuario
    args = parser.parse_args()
    
    # Usamos los valores de los argumentos, aplicando la lógica por defecto si no se especifican
    nombre = args.nombre
    edad = args.edad
    
    # Mostramos un mensaje utilizando los valores obtenidos
    print(f"Hola, {nombre}. Según la lógica por defecto, tu edad es {edad} años.")

# Verificamos si el script se está ejecutando directamente
if __name__ == "__main__":
    main()
