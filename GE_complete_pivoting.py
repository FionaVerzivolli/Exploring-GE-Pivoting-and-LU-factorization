import copy

def forward_complete(A: list[list[float]], b, rowpiv: list[int],
                      colpiv: list[int], size: int) -> list[float]:
    x = [0]*size # build x
    for i in range(size):
        tot = 0
        for j in range(i):
            tot += A[rowpiv[i]][colpiv[j]] * x[j]
        x[i] = (b[rowpiv[i]]- tot)
        
    return x

def backward_complete(A: list[list[float]], y: list[float], 
                      rowpiv: list[int], colpiv: list[int], size: int) -> list[float]:
    x = [0]*size # build x
    for i in range(size - 1, -1, -1):
        tot = 0
        for j in range(i + 1, size):
            tot += A[rowpiv[i]][colpiv[j]] * x[colpiv[j]]
        x[colpiv[i]] = (y[i] - tot) / A[rowpiv[i]][colpiv[i]]
    return x

def complete_pivot(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with complete pivoting"""
    size = len(A)
    # create separate pivot vectors to keep track of row and column interchanges
    rowpiv = [i for i in range(size)] # make row pivot
    colpiv = [i for i in range(size)] # make column pivot
    A = copy.deepcopy(A)
    # make appropriate swaps
    for n in range(size):
        max_row = n
        max_col = n
        for i in range(n, size): # search through remaining submatrix
            for j in range(n, size):
                if abs(A[rowpiv[i]][colpiv[j]]) > abs(A[rowpiv[max_row]][colpiv[max_col]]):
                    max_row = i
                    max_col = j
        # swap rows
        rowpiv[n], rowpiv[max_row] = rowpiv[max_row], rowpiv[n]
        colpiv[n], colpiv[max_col] = colpiv[max_col], colpiv[n]

        # then, do the LU decomposition with the row
        for i in range(n + 1, size): # go through rows
            if(A[rowpiv[n]][colpiv[n]] == 0):
                return
            idx = A[rowpiv[i]][colpiv[n]] / \
                A[rowpiv[n]][colpiv[n]]  # calculate coefficient
            for k in range(n, size): # update row
                A[rowpiv[i]][colpiv[k]] -= \
                    A[rowpiv[n]][colpiv[k]] * idx
            A[rowpiv[i]][colpiv[n]] = idx # fix lower matrix

    y = forward_complete(A, b, rowpiv, colpiv, size)
    x = backward_complete(A, y, rowpiv, colpiv, size)

    return x