def is_safe(board, row, col, n):
    # Verificar la columna
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Verificar la diagonal superior izquierda
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Verificar la diagonal superior derecha
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_with_backjumping(board, row, n, conflicts):
    if row >= n:
        return True

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            if solve_n_queens_with_backjumping(board, row + 1, n, conflicts):
                return True
            board[row][col] = 0
            conflicts[row] = col

    # Realizar backjumping
    if row > 0:
        conflict_column = conflicts[row - 1]
        for r in range(row - 1, -1, -1):
            if conflict_column in [board[r][c] for c in range(n) if board[r][c] == 1]:
                conflicts[r] = conflict_column
                return False

    return False

def n_queens_with_backjumping(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    conflicts = [-1] * n
    if not solve_n_queens_with_backjumping(board, 0, n, conflicts):
        print("No existe solución.")
        return None
    return board

# Ejemplo de uso
n = 4
solution = n_queens_with_backjumping(n)

if solution:
    print(f"Solución para {n}-reinas con backjumping:")
    for row in solution:
        print(row)
