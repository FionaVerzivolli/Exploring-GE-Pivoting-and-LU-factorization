"""
This file contains the implementation of the Gaussian Elimination algorithm without any pivoting.

The Gaussian Elimination algorithm is used to solve a system of linear equations of the form Ax = b.
This version of the algorithm does not perform any pivoting during the elimination process, 
which can cause instability when dealing with ill-conditioned or singular matrices, as will be 
observed during testing.

Functions:
    no_pivot(A, b): Solves the system of linear equations Ax = b using Gaussian Elimination without pivoting.
    forward_sub(A, b, size): Performs forward substitution on a lower triangular matrix A.
    back_sub(A, b, size): Performs backward substitution on an upper triangular matrix A.
    LU_factorization(A): Perform LU factorization on matrix A.
"""
import copy

def forward_sub(A: list[list[float]], b: list[float], size: int) -> list[float]:
    """return x in Ax = b on lower triangular matrix
    In other words, perform forward substitution on our triangular matrix."""

    x = [0]*size # build x
    for i in range(0, size):
        tot = 0
        for j in range(0, i):
            tot += A[i][j] * x[j]
        x[i] = (b[i]- tot) # note A[i][i] should be 1
        
    return x

def back_sub(A: list[list[float]], y: list[float], size: int) -> list[float]:
    """return x in Ax = b on upper triangular matrix
    In other words, perform backwards substitution on our triangular matrix."""
    x = [0]*size # build x

    for i in range(size - 1, -1, -1):
        tot = 0
        for j in range(i + 1, size):
            tot += A[i][j] * x[j]
        x[i] = (y[i] - tot)/A[i][i]
    return x

def no_pivot(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with no pivoting. Assume matrix is 0 indexed."""
    A = copy.deepcopy(A)
    lu = LU_factorization(A)
    # let y = Ux, Ly = b
    # use forward substitution, then backwards:
    if lu is None:
        print("Error, division by 0 along the diagonal.")
        return
    y = forward_sub(lu, b, len(A))
    x = back_sub(lu, y, len(A))
    return x # our solution to the linear system


def LU_factorization(A: list[list[float]]) -> list[list[float]]:
    """get upper and lower triangle of matrix A in one matrix.
    Helper function to perform LU factorization."""
    size = len(A)
    # make copy to avoid mutation for testing:
    a = A.copy()
    for i in range(size):
        for j in range(i + 1, size):
            if(a[i][i] == 0):
                return
            idx = a[j][i]/a[i][i] # calculate coefficient
            for k in range(i + 1, size): # update all elements of a row
                a[j][k] -= idx* a[i][k] # row subtraction
            a[j][i] = idx # replace entry with multiplier for L
    
    return a
