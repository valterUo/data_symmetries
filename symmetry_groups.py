import numpy as np
from itertools import permutations
from typing import List, Callable, Tuple

class SymmetryGroup:
    """
    A class representing a symmetry group S with matrix representations V_s.
    Each group element s acts on data vectors x via matrix multiplication V_s @ x.
    """
    
    def __init__(self, elements: List, dim: int, representation_map: Callable):
        """
        Initialize a symmetry group.
        
        Parameters:
        -----------
        elements : List
            List of group elements (can be permutations, tuples, etc.)
        dim : int
            Dimension of the vector space on which the group acts
        representation_map : Callable
            Function that maps group element s to matrix V_s
        """
        self.elements = elements
        self.dim = dim
        self.representation_map = representation_map
        self._matrices = None
    
    @property
    def matrices(self):
        """Lazy computation of all representation matrices V_s."""
        if self._matrices is None:
            self._matrices = {i: self.representation_map(s) 
                            for i, s in enumerate(self.elements)}
        return self._matrices
    
    def get_matrix(self, element_idx: int) -> np.ndarray:
        """Get the matrix V_s for group element s at index element_idx."""
        return self.matrices[element_idx]
    
    def apply(self, element_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Apply group action: compute V_s @ x
        
        Parameters:
        -----------
        element_idx : int
            Index of the group element
        x : np.ndarray
            Data vector
            
        Returns:
        --------
        np.ndarray
            Transformed data vector V_s @ x
        """
        V_s = self.get_matrix(element_idx)
        return V_s @ x
    
    def verify_group_properties(self, tolerance=1e-10) -> dict:
        """
        Verify basic group properties of the representation.
        
        Returns:
        --------
        dict with keys:
            - 'identity_exists': bool
            - 'invertible': bool (all matrices are invertible)
        """
        results = {}
        
        # Check if identity exists (at least one matrix close to identity)
        identity = np.eye(self.dim)
        has_identity = any(np.allclose(V, identity, atol=tolerance) 
                          for V in self.matrices.values())
        results['identity_exists'] = has_identity
        
        # Check if all matrices are invertible
        all_invertible = True
        for V in self.matrices.values():
            try:
                det = np.linalg.det(V)
                if abs(det) < tolerance:
                    all_invertible = False
                    break
            except np.linalg.LinAlgError:
                all_invertible = False
                break
        results['invertible'] = all_invertible
        
        return results
    
    def __len__(self):
        """Return the order of the group."""
        return len(self.elements)
    
    def __repr__(self):
        return f"SymmetryGroup(order={len(self)}, dim={self.dim})"


# ============================================================================
# Factory functions for common symmetry groups
# ============================================================================

def permutation_to_matrix(perm: Tuple[int, ...], dim: int) -> np.ndarray:
    """
    Convert a permutation to its permutation matrix representation.
    
    Parameters:
    -----------
    perm : Tuple[int, ...]
        Permutation as a tuple (0-indexed), e.g., (1, 0, 2) swaps first two elements
    dim : int
        Dimension of the matrix
        
    Returns:
    --------
    np.ndarray
        Permutation matrix V where V[i, j] = 1 if perm[j] = i, else 0
    """
    P = np.zeros((dim, dim))
    for i, j in enumerate(perm):
        P[j, i] = 1
    return P


def create_symmetric_group(n: int) -> SymmetryGroup:
    """
    Create the symmetric group S_n acting on n-dimensional vectors by permutations.
    
    Parameters:
    -----------
    n : int
        The degree of the symmetric group
        
    Returns:
    --------
    SymmetryGroup
        The symmetric group S_n with permutation matrix representations
    """
    elements = list(permutations(range(n)))
    
    def repr_map(perm):
        return permutation_to_matrix(perm, n)
    
    return SymmetryGroup(elements, n, repr_map)


def create_cyclic_group(n: int, dim: int) -> SymmetryGroup:
    """
    Create the cyclic group Z_n acting on dim-dimensional vectors by cyclic permutations.
    
    Parameters:
    -----------
    n : int
        Order of the cyclic group
    dim : int
        Dimension of the vector space (typically dim = n)
        
    Returns:
    --------
    SymmetryGroup
        The cyclic group Z_n
    """
    # Generate cyclic permutations
    elements = []
    base = list(range(dim))
    for k in range(n):
        rotated = base[k:] + base[:k]
        elements.append(tuple(rotated))
    
    def repr_map(perm):
        return permutation_to_matrix(perm, dim)
    
    return SymmetryGroup(elements, dim, repr_map)


def create_dihedral_group(n: int) -> SymmetryGroup:
    """
    Create the dihedral group D_n (symmetries of regular n-gon).
    Acts on n-dimensional vectors.
    
    Parameters:
    -----------
    n : int
        Order parameter (D_n has 2n elements)
        
    Returns:
    --------
    SymmetryGroup
        The dihedral group D_n
    """
    elements = []
    
    # Rotations: k positions for k = 0, 1, ..., n-1
    for k in range(n):
        rotated = tuple((i + k) % n for i in range(n))
        elements.append(('R', rotated))
    
    # Reflections: flip followed by rotations
    for k in range(n):
        reflected = tuple((k - i) % n for i in range(n))
        elements.append(('F', reflected))
    
    def repr_map(elem):
        _, perm = elem
        return permutation_to_matrix(perm, n)
    
    return SymmetryGroup(elements, n, repr_map)


def create_subgroup(parent_group: SymmetryGroup, 
                   element_indices: List[int]) -> SymmetryGroup:
    """
    Create a subgroup by selecting specific elements from a parent group.
    
    Parameters:
    -----------
    parent_group : SymmetryGroup
        The parent symmetry group
    element_indices : List[int]
        Indices of elements to include in the subgroup
        
    Returns:
    --------
    SymmetryGroup
        The subgroup
    """
    sub_elements = [parent_group.elements[i] for i in element_indices]
    
    return SymmetryGroup(
        sub_elements,
        parent_group.dim,
        parent_group.representation_map
    )


def create_custom_group(elements: List, 
                       dim: int,
                       representation_map: Callable) -> SymmetryGroup:
    """
    Create a custom symmetry group with user-defined elements and representation.
    
    Parameters:
    -----------
    elements : List
        List of abstract group elements
    dim : int
        Dimension of the representation space
    representation_map : Callable
        Function mapping each element to a dim x dim numpy array
        
    Returns:
    --------
    SymmetryGroup
        The custom symmetry group
    """
    return SymmetryGroup(elements, dim, representation_map)


# ============================================================================
# Utility functions
# ============================================================================

def apply_group_action_to_dataset(group: SymmetryGroup, 
                                 X: np.ndarray) -> np.ndarray:
    """
    Apply all group transformations to a dataset.
    
    Parameters:
    -----------
    group : SymmetryGroup
        The symmetry group
    X : np.ndarray
        Data matrix of shape (n_samples, dim) where dim = group.dim
        
    Returns:
    --------
    np.ndarray
        Augmented dataset of shape (n_samples * |G|, dim)
    """
    n_samples = X.shape[0]
    group_order = len(group)
    
    X_augmented = np.zeros((n_samples * group_order, group.dim))
    
    for i in range(n_samples):
        for g_idx in range(group_order):
            X_augmented[i * group_order + g_idx] = group.apply(g_idx, X[i])
    
    return X_augmented


def compute_orbit(group: SymmetryGroup, x: np.ndarray) -> np.ndarray:
    """
    Compute the orbit of a data point x under the group action.
    Orbit(x) = {V_s @ x : s ∈ S}
    
    Parameters:
    -----------
    group : SymmetryGroup
        The symmetry group
    x : np.ndarray
        Data vector of shape (dim,)
        
    Returns:
    --------
    np.ndarray
        Array of shape (|G|, dim) containing all transformed versions of x
    """
    orbit = np.zeros((len(group), group.dim))
    for i in range(len(group)):
        orbit[i] = group.apply(i, x)
    return orbit


def check_equivariance(V_s: np.ndarray, 
                      V_t: np.ndarray, 
                      V_st: np.ndarray,
                      tolerance: float = 1e-10) -> bool:
    """
    Check if V_s @ V_t = V_{st} (group homomorphism property).
    
    Parameters:
    -----------
    V_s, V_t, V_st : np.ndarray
        Matrix representations
    tolerance : float
        Numerical tolerance
        
    Returns:
    --------
    bool
        True if the equivariance holds
    """
    return np.allclose(V_s @ V_t, V_st, atol=tolerance)


def create_induced_subgroup(parent_group: SymmetryGroup, 
                           generator_indices: List[int],
                           max_iterations: int = 1000) -> SymmetryGroup:
    """
    Create the subgroup induced (generated) by a set of elements from a parent group.
    The induced subgroup contains all elements that can be obtained by composing
    the generators, including the identity.
    
    Parameters:
    -----------
    parent_group : SymmetryGroup
        The parent symmetry group
    generator_indices : List[int]
        Indices of generator elements in the parent group
    max_iterations : int
        Maximum number of iterations to prevent infinite loops
        
    Returns:
    --------
    SymmetryGroup
        The induced subgroup generated by the specified elements
        
    Notes:
    ------
    The algorithm works by repeatedly composing generators until no new elements
    are found (closure property). For finite groups, this always terminates.
    """
    dim = parent_group.dim
    tolerance = 1e-10
    
    # Get generator matrices
    generator_matrices = [parent_group.get_matrix(i) for i in generator_indices]
    
    # Start with identity (find it in parent group or create it)
    identity = np.eye(dim)
    subgroup_matrices = [identity]
    
    # Add all generators
    for gen_matrix in generator_matrices:
        # Check if not already present
        is_new = True
        for existing in subgroup_matrices:
            if np.allclose(gen_matrix, existing, atol=tolerance):
                is_new = False
                break
        if is_new:
            subgroup_matrices.append(gen_matrix)
    
    # Generate subgroup by composition until closure
    iteration = 0
    changed = True
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        current_size = len(subgroup_matrices)
        
        # Try composing each pair of elements
        for i in range(current_size):
            for j in range(current_size):
                new_matrix = subgroup_matrices[i] @ subgroup_matrices[j]
                
                # Check if this is a new element
                is_new = True
                for existing in subgroup_matrices:
                    if np.allclose(new_matrix, existing, atol=tolerance):
                        is_new = False
                        break
                
                if is_new:
                    subgroup_matrices.append(new_matrix)
                    changed = True
    
    if iteration >= max_iterations:
        print(f"Warning: Maximum iterations ({max_iterations}) reached. "
              f"Subgroup may be incomplete.")
    
    # Map subgroup matrices back to concrete permutation elements
    sub_elements = []
    parent_matrices = parent_group.matrices

    if True:
        def matrix_to_permutation(matrix: np.ndarray) -> Tuple[int, ...]:
            perm = []
            for col in range(dim):
                column = matrix[:, col]
                ones = np.where(np.isclose(column, 1.0, atol=tolerance))[0]
                if len(ones) != 1 or not np.allclose(column.sum(), 1.0, atol=tolerance):
                    raise ValueError("Encounters a matrix that is not a valid permutation matrix.")
                perm.append(int(ones[0]))
            return tuple(perm)

        for mat in subgroup_matrices:
            matched_element = None
            for idx, parent_mat in parent_matrices.items():
                if np.allclose(mat, parent_mat, atol=tolerance):
                    matched_element = parent_group.elements[idx]
                    break
            if matched_element is None:
                matched_element = matrix_to_permutation(mat)
            sub_elements.append(matched_element)
    else:
        sub_elements = list(range(len(subgroup_matrices)))  # Abstract elements as indices
    # Create representation map using stored matrices
    matrices_dict = {i: mat for i, mat in enumerate(subgroup_matrices)}
    
    def repr_map(element_idx):
        return matrices_dict[element_idx]
    
    induced_subgroup = SymmetryGroup(sub_elements, dim, repr_map)
    
    # Pre-populate the matrices cache
    induced_subgroup._matrices = matrices_dict
    
    print(f"Induced subgroup has {len(induced_subgroup)} elements "
          f"(generated from {len(generator_indices)} generators in {iteration} iterations)")
    
    return induced_subgroup


def verify_subgroup_closure(subgroup: SymmetryGroup, 
                           tolerance: float = 1e-10) -> bool:
    """
    Verify that a subgroup satisfies the closure property.
    For all elements g, h in the subgroup, g * h should also be in the subgroup.
    
    Parameters:
    -----------
    subgroup : SymmetryGroup
        The subgroup to verify
    tolerance : float
        Numerical tolerance for matrix comparison
        
    Returns:
    --------
    bool
        True if closure property holds
    """
    n = len(subgroup)
    
    for i in range(n):
        for j in range(n):
            product = subgroup.get_matrix(i) @ subgroup.get_matrix(j)
            
            # Check if product is in the subgroup
            found = False
            for k in range(n):
                if np.allclose(product, subgroup.get_matrix(k), atol=tolerance):
                    found = True
                    break
            
            if not found:
                return False
    
    return True