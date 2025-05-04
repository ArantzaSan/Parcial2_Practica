# Código para calcular una probabilidad utilizando la regla de la cadena

def probabilidad_condicional(p_a, p_b_dado_a, p_b_dado_no_a):
    """
    Calcula la probabilidad conjunta P(A y B) y la probabilidad condicional P(A | B)
    utilizando la regla de la cadena.

    Args:
        p_a (float): Probabilidad de A, P(A).
        p_b_dado_a (float): Probabilidad de B dado A, P(B | A).
        p_b_dado_no_a (float): Probabilidad de B dado no A, P(B | ¬A).

    Returns:
        tuple: (P(A y B), P(A | B))
    """
    # Probabilidad de no A
    p_no_a = 1 - p_a

    # Probabilidad total de B (teorema de probabilidad total)
    p_b = p_b_dado_a * p_a + p_b_dado_no_a * p_no_a

    # Probabilidad conjunta P(A y B)
    p_a_y_b = p_b_dado_a * p_a

    # Probabilidad condicional P(A | B) usando la regla de Bayes
    p_a_dado_b = p_a_y_b / p_b

    return p_a_y_b, p_a_dado_b


# Ejemplo de uso
p_a = 0.3  # Probabilidad de A
p_b_dado_a = 0.8  # Probabilidad de B dado A
p_b_dado_no_a = 0.2  # Probabilidad de B dado no A

p_a_y_b, p_a_dado_b = probabilidad_condicional(p_a, p_b_dado_a, p_b_dado_no_a)

print(f"P(A y B): {p_a_y_b:.4f}")
print(f"P(A | B): {p_a_dado_b:.4f}")