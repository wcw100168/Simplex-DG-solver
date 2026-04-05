"""
2D simplex orthonormal basis functions and derivatives.

Implements evaluation and differentiation of the Dubiner basis on the reference
triangle [-1, 1] × [0, 1] with collapsed coordinate mapping.
"""

import numpy as np
from ..numerics import jacobi_p, grad_jacobi_p


def evaluate_simplex_basis_2d(
    a: np.ndarray,
    b: np.ndarray,
    i: int,
    j: int
) -> np.ndarray:
    """
    Evaluate 2D simplex orthonormal basis of order (i, j) at collapsed coordinates (a, b).
    
    The 2D orthonormal basis is constructed from Jacobi polynomials as:
        psi_{i,j}(a, b) = sqrt(2) * P_i^{(0,0)}(a) * P_j^{(2i+1, 0)}(b) * (1-b)^i
    
    Parameters
    ----------
    a : np.ndarray
        First collapsed coordinate (array or scalar)
    b : np.ndarray
        Second collapsed coordinate (array or scalar)
    i : int
        First polynomial degree (i >= 0)
    j : int
        Second polynomial degree (j >= 0)
        
    Returns
    -------
    np.ndarray
        Basis function values, with shape matching input coordinates
    """
    # Evaluate Jacobi polynomials
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)

    # Combine into simplex basis
    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i


def dubiner_basis_derivative(
    a: np.ndarray,
    b: np.ndarray,
    i: int,
    j: int
) -> tuple:
    """
    Evaluate partial derivatives (dpsi_da, dpsi_db) of the 2D simplex basis in collapsed coordinates.
    
    Uses the chain rule and product rule with careful handling of the (1-b)^i singular term.
    
    Parameters
    ----------
    a : np.ndarray
        First collapsed coordinate
    b : np.ndarray
        Second collapsed coordinate
    i : int
        First polynomial degree
    j : int
        Second polynomial degree
        
    Returns
    -------
    tuple
        (dpsi_da, dpsi_db) partial derivatives in collapsed coordinate space
    """
    # Evaluate 1D polynomials and their derivatives
    p_i = jacobi_p(a, 0.0, 0.0, i)
    dp_i_da = grad_jacobi_p(a, 0.0, 0.0, i)
    
    p_j = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    dp_j_db = grad_jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    
    sqrt2 = np.sqrt(2.0)
    
    # Initialize derivatives
    dpsi_dr = np.zeros_like(a)
    dpsi_ds = np.zeros_like(a)
    
    # Applying Chain Rule and canceling out (1-b) denominator analytically
    if i == 0:
        # If i=0, dp_i_da = 0, so the 'r' derivative is identically zero.
        dpsi_dr = np.zeros_like(a)
        dpsi_ds = sqrt2 * p_i * dp_j_db
    else:
        # Partial derivative w.r.t r
        dpsi_dr = sqrt2 * dp_i_da * p_j * 2.0 * ((1.0 - b) ** (i - 1))
        
        # Partial derivative w.r.t s
        # Splitting the product rule application for clarity
        term1 = dp_i_da * p_j * (1.0 + a) * ((1.0 - b) ** (i - 1))
        term2 = p_i * dp_j_db * ((1.0 - b) ** i)
        term3 = -i * p_i * p_j * ((1.0 - b) ** (i - 1))
        
        dpsi_ds = sqrt2 * (term1 + term2 + term3)
        
    return dpsi_dr, dpsi_ds


def dubiner_basis_index_to_order(n: int) -> tuple:
    """
    Convert flattened basis index to Dubiner order pair (p, q) in O(1) time.
    
    This function is a convenience wrapper for the canonical implementation
    in the numerics.filters module. See that module for details.
    
    Parameters
    ----------
    n : int
        Flattened basis index
        
    Returns
    -------
    tuple
        Order pair (p, q)
    """
    from ..numerics import dubiner_basis_index_to_order as index_to_order
    return index_to_order(n)
