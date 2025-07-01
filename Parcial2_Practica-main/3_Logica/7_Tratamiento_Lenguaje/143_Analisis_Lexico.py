import re

# 143_Analisis_Lexico.py
# Análisis léxico básico para tratamiento de lenguaje en Python


# Definición de los patrones de los tokens (palabras clave, identificadores, números, operadores, etc.)
TOKEN_SPECIFICATION = [
    ('NUMBER',   r'\d+(\.\d*)?'),      # Números enteros o decimales
    ('ASSIGN',   r'='),                # Operador de asignación
    ('END',      r';'),                # Fin de instrucción
    ('ID',       r'[A-Za-z_]\w*'),     # Identificadores
    ('OP',       r'[+\-*/]'),          # Operadores aritméticos
    ('NEWLINE',  r'\n'),               # Saltos de línea
    ('SKIP',     r'[ \t]+'),           # Espacios y tabulaciones
    ('MISMATCH', r'.'),                # Cualquier otro carácter
]

# Compilación de los patrones en una expresión regular
token_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPECIFICATION)
get_token = re.compile(token_regex).match

# Función principal de análisis léxico
def lexer(code):
    """
    Analiza el código fuente y retorna una lista de tokens.
    """
    line_num = 1
    line_start = 0
    pos = 0
    tokens = []
    mo = get_token(code)
    while mo is not None:
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
            tokens.append(('NUMBER', value))
        elif kind == 'ID':
            tokens.append(('ID', value))
        elif kind == 'ASSIGN':
            tokens.append(('ASSIGN', value))
        elif kind == 'END':
            tokens.append(('END', value))
        elif kind == 'OP':
            tokens.append(('OP', value))
        elif kind == 'NEWLINE':
            line_num += 1
            line_start = mo.end()
        elif kind == 'SKIP':
            pass  # Ignorar espacios y tabulaciones
        elif kind == 'MISMATCH':
            raise RuntimeError(f'Caracter inesperado {value!r} en la línea {line_num}')
        pos = mo.end()
        mo = get_token(code, pos)
    return tokens

# Ejemplo de uso
if __name__ == "__main__":
    # Código de ejemplo para analizar
    code = """
    x = 10;
    y = 20;
    z = x + y * 2;
    """
    # Llamada al analizador léxico
    tokens = lexer(code)
    # Mostrar los tokens generados
    for token in tokens:
        print(token)
