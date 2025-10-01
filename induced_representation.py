import numpy as np
from typing import Callable, List, Tuple, Optional
from scipy.linalg import logm, expm
from symmetry_groups import SymmetryGroup

# ============================================================================
# Quantum State Encoding
# ============================================================================

def angle_encoding(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Angle encoding: encode classical data into a quantum state.
    |ψ(x)⟩ = ⊗_i |ψ(x_i)⟩ where |ψ(x_i)⟩ = cos(x_i)|0⟩ + sin(x_i)|1⟩
    
    Parameters:
    -----------
    x : np.ndarray
        Classical data vector of shape (n_features,)
    n_qubits : int
        Number of qubits (must be >= n_features)
        
    Returns:
    --------
    np.ndarray
        Quantum state vector of shape (2^n_qubits,)
    """
    n_features = len(x)
    if n_qubits < n_features:
        raise ValueError(f"Need at least {n_features} qubits for {n_features} features")
    
    # Start with |0⟩ state for each qubit
    state = np.array([1.0, 0.0])
    
    # Encode each feature
    for i in range(n_features):
        qubit_state = np.array([np.cos(x[i]), np.sin(x[i])])
        state = np.kron(state, qubit_state)
    
    # Pad with |0⟩ states if we have more qubits than features
    for i in range(n_features, n_qubits):
        state = np.kron(state, np.array([1.0, 0.0]))
    
    return state


def amplitude_encoding(x: np.ndarray) -> np.ndarray:
    """
    Amplitude encoding: encode classical data directly as quantum amplitudes.
    |ψ(x)⟩ = x / ||x||
    
    Parameters:
    -----------
    x : np.ndarray
        Classical data vector (will be normalized)
        
    Returns:
    --------
    np.ndarray
        Normalized quantum state vector
    """
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        raise ValueError("Cannot encode zero vector")
    return x / norm


def basis_encoding(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Basis encoding: encode classical binary data in computational basis.
    For integer index i, creates |i⟩ = |0...010...0⟩
    
    Parameters:
    -----------
    x : np.ndarray
        Classical data (will be converted to integer index)
    n_qubits : int
        Number of qubits
        
    Returns:
    --------
    np.ndarray
        Quantum state vector in computational basis
    """
    # Convert data to index
    idx = int(np.sum(x * (2 ** np.arange(len(x)))))
    dim = 2 ** n_qubits
    
    if idx >= dim:
        raise ValueError(f"Index {idx} exceeds dimension {dim}")
    
    state = np.zeros(dim)
    state[idx] = 1.0
    return state


# ============================================================================
# Deriving Unitaries U_s from Equivariance
# ============================================================================

def derive_unitaries_from_equivariance(
    X: np.ndarray,
    group: SymmetryGroup,
    encoding: Callable[[np.ndarray], np.ndarray],
    method: str = 'least_squares',
    regularization: float = 1e-6
) -> dict:
    """
    Derive unitaries U_s that satisfy the equivariance condition:
    U(V_s[x]) = U_s U(x) U_s^†
    
    This solves for U_s given:
    - A set of data points X
    - Group representations V_s
    - An encoding function U(x)
    
    Parameters:
    -----------
    X : np.ndarray
        Data points of shape (n_samples, n_features)
    group : SymmetryGroup
        The symmetry group with representations V_s
    encoding : Callable
        Function that maps data vector to quantum state: x -> U(x)|0⟩
    method : str
        Method to derive U_s: 'least_squares' or 'procrustes'
    regularization : float
        Regularization parameter for numerical stability
        
    Returns:
    --------
    dict
        Dictionary mapping group element index to unitary matrix U_s
    """
    n_samples = X.shape[0]
    
    # Get dimension of quantum state
    sample_state = encoding(X[0])
    state_dim = len(sample_state)
    
    # Encode all data points
    states = np.array([encoding(X[i]) for i in range(n_samples)])
    
    unitaries = {}
    
    for g_idx in range(len(group)):
        # Transform data points: x -> V_s[x]
        X_transformed = np.array([group.apply(g_idx, X[i]) for i in range(n_samples)])
        
        # Encode transformed data points
        states_transformed = np.array([encoding(X_transformed[i]) 
                                       for i in range(n_samples)])
        
        # Derive U_s using the specified method
        if method == 'least_squares':
            U_s = _derive_unitary_least_squares(
                states, states_transformed, regularization
            )
        elif method == 'procrustes':
            U_s = _derive_unitary_procrustes(
                states, states_transformed, regularization
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        unitaries[g_idx] = U_s
    
    return unitaries


def _derive_unitary_least_squares(
    states_in: np.ndarray,
    states_out: np.ndarray,
    regularization: float = 1e-6
) -> np.ndarray:
    """
    Derive unitary U such that U|ψ_i⟩ ≈ |φ_i⟩ using least squares.
    
    Solves: min_U Σ_i ||U|ψ_i⟩ - |φ_i⟩||^2
    subject to U being unitary
    
    Parameters:
    -----------
    states_in : np.ndarray
        Input states |ψ_i⟩, shape (n_samples, state_dim)
    states_out : np.ndarray
        Output states |φ_i⟩, shape (n_samples, state_dim)
    regularization : float
        Regularization for numerical stability
        
    Returns:
    --------
    np.ndarray
        Unitary matrix U
    """
    # Construct matrices A and B where we want U @ A ≈ B
    A = states_in.T  # shape: (state_dim, n_samples)
    B = states_out.T  # shape: (state_dim, n_samples)
    
    # Use Procrustes-like solution: U = B @ A^† @ (A @ A^†)^{-1}
    # For unitary constraint, use polar decomposition
    
    M = B @ A.conj().T
    
    # Polar decomposition: M = U @ P where U is unitary
    U, S, Vh = np.linalg.svd(M)
    U_unitary = U @ Vh
    
    return U_unitary


def _derive_unitary_procrustes(
    states_in: np.ndarray,
    states_out: np.ndarray,
    regularization: float = 1e-6
) -> np.ndarray:
    """
    Derive unitary using orthogonal Procrustes problem.
    
    Solves: min_U ||U @ A - B||_F subject to U^† U = I
    
    Parameters:
    -----------
    states_in : np.ndarray
        Input states, shape (n_samples, state_dim)
    states_out : np.ndarray
        Output states, shape (n_samples, state_dim)
    regularization : float
        Regularization for numerical stability
        
    Returns:
    --------
    np.ndarray
        Unitary matrix U
    """
    A = states_in.T
    B = states_out.T
    
    # Compute cross-covariance matrix
    M = B @ A.conj().T
    
    # SVD of M
    U, S, Vh = np.linalg.svd(M)
    
    # Optimal unitary is U @ V^†
    U_optimal = U @ Vh
    
    # Ensure determinant is +1 (proper rotation)
    if np.linalg.det(U_optimal) < 0:
        U[:, -1] *= -1
        U_optimal = U @ Vh
    
    return U_optimal


# ============================================================================
# Verification and Utilities
# ============================================================================

def verify_equivariance(
    X: np.ndarray,
    group: SymmetryGroup,
    encoding: Callable[[np.ndarray], np.ndarray],
    unitaries: dict,
    tolerance: float = 1e-4
) -> Tuple[bool, dict]:
    """
    Verify that the derived unitaries satisfy equivariance:
    U(V_s[x]) = U_s U(x) U_s^†
    
    Parameters:
    -----------
    X : np.ndarray
        Data points of shape (n_samples, n_features)
    group : SymmetryGroup
        The symmetry group
    encoding : Callable
        Encoding function
    unitaries : dict
        Dictionary of derived unitaries
    tolerance : float
        Numerical tolerance for verification
        
    Returns:
    --------
    Tuple[bool, dict]
        (all_passed, detailed_results)
    """
    n_samples = X.shape[0]
    results = {}
    all_passed = True
    
    for g_idx in range(len(group)):
        U_s = unitaries[g_idx]
        errors = []
        
        for i in range(n_samples):
            # Left side: U(V_s[x])
            x_transformed = group.apply(g_idx, X[i])
            left_side = encoding(x_transformed)
            
            # Right side: U_s U(x) U_s^†
            state_x = encoding(X[i])
            # Note: U_s acts on the state, not on the density matrix
            # So we have U_s |ψ(x)⟩
            right_side = U_s @ state_x
            
            # Compute error
            error = np.linalg.norm(left_side - right_side)
            errors.append(error)
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        passed = max_error < tolerance
        
        results[g_idx] = {
            'mean_error': mean_error,
            'max_error': max_error,
            'passed': passed
        }
        
        if not passed:
            all_passed = False
    
    return all_passed, results


def check_unitary(U: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if a matrix is unitary: U^† U = I
    
    Parameters:
    -----------
    U : np.ndarray
        Matrix to check
    tolerance : float
        Numerical tolerance
        
    Returns:
    --------
    bool
        True if matrix is unitary
    """
    identity = np.eye(len(U))
    product = U.conj().T @ U
    return np.allclose(product, identity, atol=tolerance)


def compute_equivariance_error(
    x: np.ndarray,
    V_s: np.ndarray,
    U_s: np.ndarray,
    encoding: Callable[[np.ndarray], np.ndarray]
) -> float:
    """
    Compute the equivariance error for a single data point:
    ||U(V_s[x]) - U_s U(x)||
    
    Parameters:
    -----------
    x : np.ndarray
        Data point
    V_s : np.ndarray
        Classical symmetry representation
    U_s : np.ndarray
        Quantum unitary
    encoding : Callable
        Encoding function
        
    Returns:
    --------
    float
        Equivariance error
    """
    # Left side: U(V_s[x])
    x_transformed = V_s @ x
    left_side = encoding(x_transformed)
    
    # Right side: U_s U(x)
    state_x = encoding(x)
    right_side = U_s @ state_x
    
    return np.linalg.norm(left_side - right_side)


def visualize_unitary_spectrum(unitaries: dict) -> dict:
    """
    Compute eigenvalues of each unitary (should lie on unit circle).
    
    Parameters:
    -----------
    unitaries : dict
        Dictionary of unitary matrices
        
    Returns:
    --------
    dict
        Dictionary mapping group element index to eigenvalues
    """
    spectra = {}
    for g_idx, U_s in unitaries.items():
        eigenvalues = np.linalg.eigvals(U_s)
        spectra[g_idx] = eigenvalues
    return spectra