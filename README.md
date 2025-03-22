# Exploring-GE-Pivoting-and-LU-factorization
# Gaussian Elimination Algorithms

This repository includes three different implementations of the Gaussian Elimination algorithm, each having different pivoting strategies. The algorithms in each file are inspired by the teachings of the CSC336 course (Numerical Methods) offered at the University of Toronto, and are based off of chapter 2 of Michael Heath's *Scientific Computing* textbook, titled *Systems of Linear Equations*. The repository includes unit tests to test correctness and functionality, as well as visualizations made using Pandas to analyze the performance of each pivoting method.

## Overview

The project includes three variants of the Gaussian Elimination algorithm:
- **No Pivoting**: The simplest version of Gaussian Elimination, which does not perform any pivoting during the elimination process.
- **Partial Pivoting**: This method selects the largest element in each column underneath the current row to use as the pivot, improving stability.
- **Complete Pivoting**: The most stable version, selecting the largest element in the entire uneliminated matrix (both row and column) for pivoting.

The repository also includes:
- **Unit tests** for validating the correctness of each algorithm and the way different matrix types are handled (ill conditioned, sparse, etc).
- **Visualization tools** to track and display performance metrics such as residuals, conditioning numbers, and errors.

## Algorithms

### 1. No Pivoting (`GE_no_pivot.py`)
This file implements Gaussian Elimination without any pivoting. This method is computationally less expensive but can suffer from numerical instability, especially when solving ill conditioned or singular systems.

### 2. Partial Pivoting (`GE_partial_pivoti.py`)
In this version, the algorithm selects the largest element in each column as the pivot. This helps to improve numerical stability, especially for ill conditioned matrices. To reduce computational load I will be using a pivot vector to keep track of row swaps.

### 3. Complete Pivoting (`GE_complete_pivoting.py`)
Complete pivoting selects the largest element in the entire uneliminated matrix (both row and column) as the pivot. This method offers the highest stability but requires more computational resources than partial pivoting, which is why I am using two pivot vectors, one for keeping track of column swaps and one for keeping track of row swaps.

