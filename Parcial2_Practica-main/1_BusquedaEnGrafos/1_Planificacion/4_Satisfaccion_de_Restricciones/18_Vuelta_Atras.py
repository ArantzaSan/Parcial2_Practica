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

def solve_n_queens(board, row, n):
    if row >= n:
        return True

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            if solve_n_queens(board, row + 1, n):
                return True
            board[row][col] = 0

    return False

def n_queens(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    if not solve_n_queens(board, 0, n):
        print("No existe solución.")
        return None
    return board

# Ejemplo de uso
n = 4
solution = n_queens(n)

if solution:
    print(f"Solución para {n}-reinas:")
    for row in solution:
        print(row)
