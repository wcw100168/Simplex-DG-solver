"""
Geometry module: Coordinate systems, mappings, and data structures.
"""

from .data_structs import Node, get_orbit, bary_to_cartesian_2d, bary_to_cartesian_3d
from .mappings import collapsed_coords_transform

__all__ = [
    "Node",
    "get_orbit",
    "bary_to_cartesian_2d",
    "bary_to_cartesian_3d",
    "collapsed_coords_transform"
]
