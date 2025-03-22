"""
This file provides functions to visualize and analyze the performance of Gaussian Elimination algorithms.

The main goal of this module is to track and display the residuals, conditioning numbers, and errors for 
the various Gaussian Elimination methods (no pivoting, partial pivoting, and complete pivoting) using a
 variety of matrices. Results are presented in a table format and a bar chart to compare performance across algorithms.

Functions:
    two_norm(a): Computes the euclidean (L2) norm of a vector.
    vector_subtraction(a, b): Computes the element wise subtraction of two vectors.
    matrix_multiply(A, x): Performs matrix vector multiplication.
    visualize_results(): Displays a table with test results and a bar chart comparing errors of different Gaussian Elimination methods.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        self.results = []
    
    def test_known_matrices(self):
        """tests Gaussian elimination methods with predefined matrices and logs results."""
        for A, b in self.test_cases:
            A, b = np.array(A, dtype=float), np.array(b, dtype=float)
            if np.linalg.det(A) != 0:  # only test if A is non-singular
                for solver in [no_pivot, partial_pivot, complete_pivot]:
                    with self.subTest(solver=solver):
                        x = solver(A, b)
                        error = two_norm(vector_subtraction(matrix_multiply(A, x), b))
                        self.results.append({"Method": solver.__name__, "Matrix Size": A.shape[0], "Error": error})

    def tearDown(self):
        """display results in a table after tests using pandas."""
        df = pd.DataFrame(self.results)
        print(df)
        df.plot(kind='bar', x='Method', y='Error', title='Error Comparison', legend=False)
        plt.ylabel('Error')
        plt.show()

if __name__ == "__main__":
    unittest.main()
