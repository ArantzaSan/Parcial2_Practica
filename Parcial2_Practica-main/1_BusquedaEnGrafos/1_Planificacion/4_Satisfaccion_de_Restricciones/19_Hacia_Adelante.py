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

def forward_checking(board, row, n, domains):
    if row >= n:
        return True

    for col in domains[row]:
        if is_safe(board, row, col, n):
            board[row][col] = 1
            next_domains = domains.copy()
            for i in range(row + 1, n):
                if col in next_domains[i]:
                    next_domains[i].remove(col)
                if (row - i >= 0 and col - (row - i) in next_domains[i]):
                    next_domains[i].remove(col - (row - i))
                if (row - i >= 0 and col + (row - i) < n and col + (row - i) in next_domains[i]):
                    next_domains[i].remove(col + (row - i))

            if all(next_domains[i] for i in range(row + 1, n)):
                if forward_checking(board, row + 1, n, next_domains):
                    return True
            board[row][col] = 0

    return False

def n_queens_forward_checking(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    domains = [list(range(n)) for _ in range(n)]
    if not forward_checking(board, 0, n, domains):
        print("No existe solución.")
        return None
    return board

# Ejemplo de uso
n = 4
solution = n_queens_forward_checking(n)

if solution:
    print(f"Solución para {n}-reinas con forward checking:")
    for row in solution:
        print(row)
