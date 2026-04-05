"""
Coordinate transformations and geometric mappings.

Implements transformations between reference element coordinates and collapsed
coordinates used in spectral element methods.
"""

import numpy as np


def collapsed_coords_transform(xi: np.ndarray, eta: np.ndarray) -> tuple:
    """
    Map reference triangle coordinates (xi, eta) to collapsed coordinates (a, b).
    
    The reference triangle is [-1, 1] × [0, 1] in (xi, eta) space.
    Collapsed coordinates (a, b) correspond to the Jacobi polynomial domain.
    
    Transformation:
        a = 2(1 + xi) / (1 - eta) - 1
        b = eta
    
    Parameters
    ----------
    xi : np.ndarray
        Reference element coordinate (array or scalar)
    eta : np.ndarray
        Reference element coordinate (array or scalar)
        
    Returns
    -------
    tuple
        (a, b) collapsed coordinates
    """
    a = np.empty_like(xi, dtype=float)
    mask = np.abs(1.0 - eta) > 1e-15
    a[mask] = 2.0 * (1.0 + xi[mask]) / (1.0 - eta[mask]) - 1.0
    a[~mask] = -1.0

    b = eta.copy()
    return a, b
