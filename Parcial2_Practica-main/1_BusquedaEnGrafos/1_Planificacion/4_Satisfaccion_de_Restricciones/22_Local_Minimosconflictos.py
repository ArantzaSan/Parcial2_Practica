import random

def count_conflicts(board, row, col, n):
    conflicts = 0
    # Verificar la columna
    for i in range(n):
        if i != row and board[i][col] == 1:
            conflicts += 1

    # Verificar la diagonal superior izquierda
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            conflicts += 1
        i -= 1
        j -= 1

    # Verificar la diagonal superior derecha
    i, j = row - 1, col + 1
    while i >= 0 and j < n:
        if board[i][j] == 1:
            conflicts += 1
        i -= 1
        j += 1

    return conflicts

def min_conflicts(board, n, max_steps=1000):
    # Inicializar el tablero con una configuración aleatoria
    for i in range(n):
        board[i][random.randint(0, n - 1)] = 1

    for _ in range(max_steps):
        # Seleccionar una reina con conflictos
        conflict_queens = [(r, c) for r in range(n) for c in range(n) if board[r][c] == 1 and count_conflicts(board, r, c, n) > 0]
        if not conflict_queens:
            return board

        row, col = random.choice(conflict_queens)
        board[row][col] = 0

        # Encontrar la columna con el mínimo conflicto para la reina
        min_conflict_col = min(range(n), key=lambda c: count_conflicts(board, row, c, n))
        board[row][min_conflict_col] = 1

    return board

def n_queens_min_conflicts(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    solution = min_conflicts(board, n)
    return solution

# Ejemplo de uso
n = 4
solution = n_queens_min_conflicts(n)

if solution:
    print(f"Solución para {n}-reinas con mínimos conflictos:")
    for row in solution:
        print(row)
