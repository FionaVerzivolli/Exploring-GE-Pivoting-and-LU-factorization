"""
This file contains the implementation of the Gaussian Elimination algorithm with partial pivoting.

Partial pivoting includes selecting the largest element in the column below our current row to use as the pivot 
during the elimination process. This improves numerical stability, especially when working with matrices that
 are ill-conditioned.

Functions:
    partial_pivot(A, b): Solves the system of linear equations Ax = b using Gaussian Elimination with partial pivoting.
    forward_partial(A, b, piv, size): Performs forward substitution on a lower triangular matrix A.
    backward_partial(A, y, piv, size): Performs backward substitution on an upper triangular matrix A.

Note that there is no LU factorization function in this file as it is dealt with in the partial_pivot function.
"""
import copy

def forward_partial(A: list[list[float]], b: list[float], piv: list[int], size: int) -> list[float]:
    """
    Return x in Ax = b on lower triangular matrix. Same as 
    forward_sub for no pivoting, except I need to use the 
    pivot vector when dealing with A and b to account for 
    swapped rows to reduce computations.
    """
    x = [0]*size # build x
    for i in range(0, size):
        tot = 0
        for j in range(0, i):
            tot += A[piv[i]][j] * x[j]
        x[i] = (b[piv[i]]- tot)
        
    return x


def backward_partial(A: list[list[float]], y: list[float], piv: list[int], size: int) -> list[float]:
    """
    Return x in Ax = b on upper triangular matrix.
    Same as backward_partial for no pivoting, except I need to use
    the pivot vector when I am dealing with A to account for
    swapped rows.
    """
    x = [0]*size # build x

    for i in range(size - 1, -1, -1):
        sum = 0
        for j in range(i + 1, size):
            sum += A[piv[i]][j] * x[j]
        x[i] = (y[i] - sum)/A[piv[i]][i]
    return x

def partial_pivot(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Assume matrix is 0 indexed."""
    size = len(A)
    # first, build our pivot vector.
    piv = [i for i in range(size)]
    A = copy.deepcopy(A)
    
    # make appropriate swaps
    for i in range(size):
        maxRow = i
        for j in range(i + 1, size):
            # swap rows with the row with the greatest abs value in the column
            if abs(A[j][i]) > abs(A[maxRow][i]):
                # update the row with maximum abs value
                maxRow = j
        # then swap rows
        piv[i], piv[maxRow] = piv[maxRow], piv[i]
        # then, do the LU decomposition with the row
        for j in range(i + 1, size):
            if(A[piv[i]][i] == 0):
                return
            idx = A[piv[j]][i] / A[piv[i]][i] # calculate coefficient
            for n in range(i, size): # update row
                A[piv[j]][n] -= A[piv[i]][n] * idx
            A[piv[j]][i] = idx # fix lower matrix
            
    y = forward_partial(A, b, piv, len(A)) # start with forward substitution,
    x = backward_partial(A, y, piv, len(A)) # then backward substitution.

    return x # our solution to the linear system
