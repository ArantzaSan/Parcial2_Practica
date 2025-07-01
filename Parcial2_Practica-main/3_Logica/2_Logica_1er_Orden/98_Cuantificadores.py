# Programa que utiliza lógica de primer orden con cuantificadores en el campo de la química

# Definimos un conjunto de elementos químicos y sus propiedades
elementos = [
    {"nombre": "Hidrógeno", "simbolo": "H", "es_gas": True, "es_no_metal": True},
    {"nombre": "Oxígeno", "simbolo": "O", "es_gas": True, "es_no_metal": True},
    {"nombre": "Hierro", "simbolo": "Fe", "es_gas": False, "es_no_metal": False},
    {"nombre": "Cloro", "simbolo": "Cl", "es_gas": True, "es_no_metal": True},
    {"nombre": "Sodio", "simbolo": "Na", "es_gas": False, "es_no_metal": False},
]

# Función que implementa el cuantificador universal (∀)
# Verifica si todos los elementos cumplen una propiedad dada
def todos_cumplen(propiedad):
    return all(propiedad(elemento) for elemento in elementos)

# Función que implementa el cuantificador existencial (∃)
# Verifica si al menos un elemento cumple una propiedad dada
def existe_al_menos_uno(propiedad):
    return any(propiedad(elemento) for elemento in elementos)

# Ejemplo 1: ¿Todos los elementos son gases?
todos_son_gases = todos_cumplen(lambda e: e["es_gas"])
print(f"¿Todos los elementos son gases? {'Sí' if todos_son_gases else 'No'}")

# Ejemplo 2: ¿Existe al menos un elemento que sea un no metal?
existe_no_metal = existe_al_menos_uno(lambda e: e["es_no_metal"])
print(f"¿Existe al menos un elemento que sea un no metal? {'Sí' if existe_no_metal else 'No'}")

# Ejemplo 3: ¿Todos los elementos que son gases son no metales?
todos_gases_son_no_metales = todos_cumplen(lambda e: not e["es_gas"] or e["es_no_metal"])
print(f"¿Todos los elementos que son gases son no metales? {'Sí' if todos_gases_son_no_metales else 'No'}")

# Ejemplo 4: ¿Existe al menos un elemento que no sea gas y no sea metal?
existe_no_gas_no_metal = existe_al_menos_uno(lambda e: not e["es_gas"] and e["es_no_metal"])
print(f"¿Existe al menos un elemento que no sea gas y no sea metal? {'Sí' if existe_no_gas_no_metal else 'No'}")
