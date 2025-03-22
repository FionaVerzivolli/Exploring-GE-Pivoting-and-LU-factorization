import unittest
import numpy as np

# import Gaussian elimination functions

from GE_no_pivoting import gaussian_elimination as no_pivot
from GE_partial_pivoting import gaussian_elimination as partial_pivot
from GE_complete_pivoting import gaussian_elimination as complete_pivot


# helper functions

def two_norm(a):
    """computes the euclidean (L2) norm of a vector."""
    return np.linalg.norm(a, 2)

def vector_subtraction(a, b):
    """computes the element-wise subtraction of two vectors."""
    return np.subtract(a, b)

def matrix_multiply(A, x):
    """performs matrix vector multiplication."""
    return np.dot(A, x)

# unit test class for Gaussian Elimination algorithms
class TestGaussianElimination(unittest.TestCase):
    def setUp(self):
        """defines test cases for Gaussian elimination methods."""
        self.test_cases = [
            ([[2, 4, -2], [4, 14, 0], [-1, 10, 7]], [4, 18, 15]),
            ([[0,2,5],[4,5,6],[7,8,9]], [1, 2, 3]),
            ([[2,4,6], [2,0,7],[1,0,0]], [1, 2, 3]),
            ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [1, 1, 1]),
        ]
    
    def test_known_matrices(self):
        """tests Gaussian elimination methods with predefined matrices."""
        for A, b in self.test_cases:
            A, b = np.array(A, dtype=float), np.array(b, dtype=float)
            if np.linalg.det(A) != 0:  # Only test if A is non-singular
                for solver in [no_pivot, partial_pivot, complete_pivot]:
                    with self.subTest(solver=solver):
                        x = solver(A, b)
                        np.testing.assert_allclose(matrix_multiply(A, x), b, rtol=1e-5)
    
    def test_random_matrices(self):
        """tests Gaussian elimination methods with randomly generated matrices."""
        num_tests, matrix_size = 100, 20
        for _ in range(num_tests):
            A = np.random.rand(matrix_size, matrix_size)
            x_original = np.array([(-1)**i for i in range(matrix_size)])
            b = matrix_multiply(A, x_original)
            for solver in [no_pivot, partial_pivot, complete_pivot]:
                with self.subTest(solver=solver):
                    x = solver(A, b)
                    error = two_norm(vector_subtraction(x, x_original))
                    self.assertLess(error, 1e-5)  # ensure small error
    
    def test_special_matrix(self):
        """tests Gaussian elimination methods with a structured matrix of size 200x200."""
        n = 200
        A = np.eye(n)
        A[:, -1] = 1
        for i in range(n):
            for j in range(i):
                A[i, j] = -1
        b = np.random.randint(0, 50, size=n)
        for solver in [partial_pivot, complete_pivot]:
            with self.subTest(solver=solver):
                x = solver(A, b)
                residual = two_norm(vector_subtraction(matrix_multiply(A, x), b))
                self.assertLess(abs(residual / two_norm(b)), 1e-5)
    
    def test_ill_conditioned_matrix(self):
        """tests Gaussian elimination methods with an ill conditioned matrix."""
        A = np.array([[1, 1], [1, 1.0001]])
        b = np.array([2, 2.0001])
        for solver in [no_pivot, partial_pivot, complete_pivot]:
            with self.subTest(solver=solver):
                x = solver(A, b)
                residual = two_norm(vector_subtraction(matrix_multiply(A, x), b))
                self.assertLess(abs(residual / two_norm(b)), 1e-3)
    
    def test_singular_matrix(self):
        """ensures Gaussian elimination methods handle singular matrices appropriately."""
        A = np.array([[1, 2], [2, 4]])  # singular matrix
        b = np.array([3, 6])
        for solver in [no_pivot, partial_pivot, complete_pivot]:
            with self.subTest(solver=solver):
                with self.assertRaises(np.linalg.LinAlgError):
                    solver(A, b)
    
    def test_sparse_matrix(self):
        """tests Gaussian elimination methods with a sparse matrix."""
        size = 50
        A = np.zeros((size, size))
        np.fill_diagonal(A, 1)
        A[0, -1] = 1
        b = np.random.rand(size)
        for solver in [partial_pivot, complete_pivot]:
            with self.subTest(solver=solver):
                x = solver(A, b)
                residual = two_norm(vector_subtraction(matrix_multiply(A, x), b))
                self.assertLess(abs(residual / two_norm(b)), 1e-5)
    
    def test_large_random_matrix(self):
        """tests Gaussian elimination methods with a large random matrix (500x500)."""
        size = 500
        A = np.random.rand(size, size)
        x_original = np.random.rand(size)
        b = matrix_multiply(A, x_original)
        for solver in [partial_pivot, complete_pivot]:
            with self.subTest(solver=solver):
                x = solver(A, b)
                error = two_norm(vector_subtraction(x, x_original))
                self.assertLess(error, 1e-5)

if __name__ == "__main__":
    unittest.main()