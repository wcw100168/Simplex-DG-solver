"""
Node generator classes for quadrature rules (Table 1 and Table 2).
"""

import numpy as np

from .data_structs import Node, get_orbit, bary_to_cartesian_2d, bary_to_cartesian_3d


class BaseNodeGenerator:
    """
    Abstract base class for node generation strategies.
    
    Parameters:
    -----------
    k : int
        Polynomial degree / basis parameter
    vertices_2d : np.ndarray
        2D reference triangle vertices (shape: [3, 2])
    vertices_3d : np.ndarray
        3D reference triangle vertices (shape: [3, 3])
    """
    
    def __init__(self, k: int, vertices_2d: np.ndarray, vertices_3d: np.ndarray):
        self.k = int(k)
        self.vertices_2d = np.asarray(vertices_2d, dtype=float)
        self.vertices_3d = np.asarray(vertices_3d, dtype=float)

    def generate(self):
        """Generate nodes. Must be implemented by subclasses."""
        raise NotImplementedError


class Table1NodeGenerator(BaseNodeGenerator):
    """
    Generate quadrature nodes based on Table 1 (exact integration rules).
    
    Supports polynomial degrees k=1,2,3,4.
    """
    
    QUADRATURE_RULES = {
        1: [
            (0.2113248654051871, 0.0000000000000000, 0.16666666666666667)
        ],
        2: [
            (0.1127016653792583, 0.0000000000000000, 0.04166666666666666),
            (0.5000000000000000, 0.0000000000000000, 0.09999999999999999),
            (0.3333333333333333, 0.3333333333333333, 0.45000000000000000)
        ],
        3: [
            (0.06943184420297367, 0.0000000000000000, 0.01509901487256561),
            (0.3300094782075718, 0.0000000000000000, 0.04045654068298990),
            (0.5841571139756568, 0.1870738791912763, 0.11111111111111111)
        ],
        4: [
            (0.04691007703066797, 0.0000000000000000, 0.006601315081001592),
            (0.2307653449471584, 0.0000000000000000, 0.02053045968042892),
            (0.5000000000000000, 0.0000000000000000, 0.01853708483394990),
            (0.1394337314154536, 0.1394337314154536, 0.10542932962084440),
            (0.4384239524408185, 0.4384239524408185, 0.12473673228977350),
            (0.3333333333333333, 0.3333333333333333, 0.09109991119771331)
        ]
    }

    def generate(self):
        """
        Generate nodes from Table 1 data.
        
        Returns:
        --------
        list of Node
            List of generated quadrature nodes
        """
        if self.k not in self.QUADRATURE_RULES:
            raise ValueError(f"Table 1 data for k={self.k} is not provided.")

        nodes = []
        node_id = 1

        for b1, b2, wt in self.QUADRATURE_RULES[self.k]:
            for bary in get_orbit(b1, b2):
                bary_mapped = np.array(bary, dtype=float)
                local = np.array([bary_mapped[2], bary_mapped[0]], dtype=float)
                g3 = bary_to_cartesian_3d(bary_mapped, self.vertices_3d)
                nodes.append(Node(node_id, bary_mapped, float(wt), local, g3))
                node_id += 1

        return nodes


class Table2NodeGenerator(BaseNodeGenerator):
    """
    Generate quadrature nodes based on Table 2 (interior nodal rules).
    
    Supports polynomial degrees k=1,2,3,4.
    """
    
    QUADRATURE_RULES = {
        1: [(0.1666666666666666, 0.1666666666666666, 0.3333333333333333)],
        2: [
            (0.09157621350977067, 0.09157621350977067, 0.1099517436553218),
            (0.4459484909159648, 0.4459484909159648, 0.2233815896780115),
        ],
        3: [
            (0.219429982549783, 0.219429982549783, 0.1713331241529809),
            (0.480137964112215, 0.480137964112215, 0.08073108959303095),
            (0.1416190159239682, 0.0193717243612408, 0.04063455979366068),
        ],
        4: [
            (0.7284923929554044, 0.2631128296346379, 0.02723031417443505),
            (0.4592925882927232, 0.4592925882927232, 0.09509163426728455),
            (0.1705693077517602, 0.1705693077517602, 0.1032173705347182),
            (0.05054722831703096, 0.05054722831703096, 0.03245849762319804),
            (0.3333333333333333, 0.3333333333333333, 0.1443156076777874),
        ],
    }

    @staticmethod
    def octa_face_map(u: float, v: float, w: float):
        """
        Map from normalized octahedron coordinates to [-1, 1]^3.
        
        Parameters:
        -----------
        u, v, w : float
            Normalized octahedron face coordinates
        
        Returns:
        --------
        np.ndarray
            Mapped coordinates x, y, z in [-1, 1]
        """
        den = (1.0 - u + v + w)
        eps = 1e-12
        if abs(den) < eps:
            den = eps if den >= 0 else -eps
        x = 2.0 * (1.0 + u) / den - 1.0
        y = 2.0 * (1.0 + v) / den - 1.0
        z = 2.0 * (1.0 + w) / den - 1.0
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def mapped_to_bary(mapped_xyz: np.ndarray):
        """
        Convert mapped coordinates back to barycentric coordinates.
        
        Parameters:
        -----------
        mapped_xyz : np.ndarray
            Coordinates in [-1, 1]^3
        
        Returns:
        --------
        np.ndarray
            Normalized barycentric coordinates
        """
        xyz01 = (mapped_xyz + 1.0) / 2.0
        xyz01 = np.clip(xyz01, 1e-12, None)
        return xyz01 / np.sum(xyz01)

    def generate(self):
        """
        Generate nodes from Table 2 data.
        
        Returns:
        --------
        list of Node
            List of generated quadrature nodes
        """
        if self.k not in self.QUADRATURE_RULES:
            raise ValueError(f"Table 2 data for k={self.k} is not provided.")

        nodes = []
        node_id = 1

        for b1, b2, wt in self.QUADRATURE_RULES[self.k]:
            for bary in get_orbit(b1, b2):
                bary_mapped = np.array(bary, dtype=float)
                local = np.array([bary_mapped[2], bary_mapped[0]], dtype=float)
                g3 = bary_to_cartesian_3d(bary_mapped, self.vertices_3d)
                nodes.append(Node(node_id, bary_mapped, float(wt), local, g3))
                node_id += 1

        return nodes


def build_nodes(method: str, k: int, vertices_2d: np.ndarray, vertices_3d: np.ndarray):
    """
    Factory function to build nodes using the specified method.
    
    Parameters:
    -----------
    method : str
        Either "table1" or "table2"
    k : int
        Polynomial degree (1-4)
    vertices_2d : np.ndarray
        2D reference triangle vertices (shape: [3, 2])
    vertices_3d : np.ndarray
        3D reference triangle vertices (shape: [3, 3])
    
    Returns:
    --------
    list of Node
        List of generated nodes
    """
    method = method.lower().strip()
    if method == "table1":
        gen = Table1NodeGenerator(k, vertices_2d, vertices_3d)
    elif method == "table2":
        gen = Table2NodeGenerator(k, vertices_2d, vertices_3d)
    else:
        raise ValueError("method must be 'table1' or 'table2'")
    return gen.generate()


def get_extra_bary(method: str, k: int):
    """
    Generate additional barycentric points for boundary/symmetry correction.
    
    Injects symmetric corrective points based on the method and degree.
    
    Parameters:
    -----------
    method : str
        Either "table1" or "table2"
    k : int
        Polynomial degree
    
    Returns:
    --------
    np.ndarray
        Array of barycentric coordinates (shape: [n_extra, 3])
    """
    extra = []
    if method == 'table1':
        if k == 1:
            extra.append([1 / 3, 1 / 3, 1 / 3])
        elif k == 3:
            extra.append([1 / 3, 1 / 3, 1 / 3])
            extra.extend(get_orbit(0.7573626348863415, 0.1213186825568292))
            extra.extend(get_orbit(0.09353693959563815, 0.4532315302021809))
    elif method == 'table2':
        if k == 1:
            extra.extend(get_orbit(11 / 24, 11 / 24))

    return np.array(extra, dtype=float) if extra else np.empty((0, 3), dtype=float)
