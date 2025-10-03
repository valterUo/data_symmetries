import numpy as np
from typing import Callable, Tuple
from twirler.symmetry_groups import SymmetryGroup

def build_qubit_permutation_unitary(perm: tuple[int, ...]) -> np.ndarray:
    """
    Return the 2^n x 2^n unitary implementing the qubit permutation 'perm',
    where perm[i] = target position of qubit i.
    Maps |b_0 b_1 ... b_{n-1}> -> |b_{p^{-1}(0)} b_{p^{-1}(1)} ...>.
    We define action so that bit originally at position i moves to position perm[i].
    """
    n = len(perm)
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    # Precompute inverse permutation to map basis indices cleanly
    inv = [0]*n
    for i, p in enumerate(perm):
        inv[p] = i
    for basis_in in range(dim):
        bits = [(basis_in >> k) & 1 for k in range(n)]
        # Reorder bits according to inverse so that new position j gets old bit from inv[j]
        permuted_bits = [bits[inv[j]] for j in range(n)]
        basis_out = sum(permuted_bits[k] << k for k in range(n))
        U[basis_out, basis_in] = 1.0
    return U

def derive_unitaries_angle_embedding_analytic(group) -> dict[int, np.ndarray]:
    """
    Analytic induced representation for permutation group acting on indices of AngleEmbedding.
    Assumes group.elements is an iterable of permutations as tuples (p0,p1,...,p_{n-1}),
    where element maps index i -> perm[i].
    """
    unitaries = {}
    for idx, perm in enumerate(group.elements):
        U_s = build_qubit_permutation_unitary(perm)
        unitaries[idx] = U_s
    return unitaries

# In induced_representation.py, modify solve_for_induced_unitary_optimization:

def solve_for_induced_unitary_optimization(
    data_points: np.ndarray,
    V_s: np.ndarray,
    encoding_unitary: callable,
    dim: int,
    verbose: bool = True
):
    """Improved optimization with better initialization and constraints."""
    from scipy.optimize import minimize
    from scipy.linalg import expm
    
    data_points = np.atleast_2d(data_points)
    
    # Try multiple initializations
    best_U_s = None
    best_error = np.inf
    
    for trial in range(5):  # Try 5 random starts
        # Initialize closer to permutation matrix
        if trial == 0:
            # Start with identity
            H_init = np.zeros((dim, dim), dtype=complex)
        else:
            # Random initialization
            H_init = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) * 0.1
            H_init = (H_init + H_init.conj().T) / 2  # Make Hermitian
        
        # Parametrize Hermitian matrix
        def pack_hermitian(H):
            params = []
            for i in range(dim):
                params.append(H[i, i].real)
                for j in range(i+1, dim):
                    params.append(H[i, j].real)
                    params.append(H[i, j].imag)
            return np.array(params)
        
        def unpack_hermitian(params):
            H = np.zeros((dim, dim), dtype=complex)
            idx = 0
            for i in range(dim):
                H[i, i] = params[idx]
                idx += 1
                for j in range(i+1, dim):
                    H[i, j] = params[idx] + 1j * params[idx+1]
                    H[j, i] = params[idx] - 1j * params[idx+1]
                    idx += 2
            return H
        
        def objective(params):
            H = unpack_hermitian(params)
            U_s = expm(1j * H)
            
            total_error = 0
            for x in data_points:
                U_x = encoding_unitary(x)
                U_Vsx = encoding_unitary(V_s @ x)
                U_conjugated = U_s @ U_x @ U_s.conj().T
                total_error += np.linalg.norm(U_Vsx - U_conjugated, 'fro')**2
            
            return total_error
        
        params_init = pack_hermitian(H_init)
        
        result = minimize(
            objective, 
            params_init, 
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            H_best = unpack_hermitian(result.x)
            best_U_s = expm(1j * H_best)
    
    if verbose:
        print(f"Best error after {5} trials: {best_error:.2e}")
    
    return best_U_s

# ============================================================================
# Deriving Unitaries U_s from Equivariance
# ============================================================================

def derive_unitaries_from_equivariance(
    X: np.ndarray,
    group: SymmetryGroup,
    encoding_unitary: Callable[[np.ndarray], np.ndarray],  # Returns UNITARY not state,
    n_qubits: int,
    method: str = 'optimization'
) -> dict:
    """
    Derive unitaries U_s that satisfy:
    U(V_s[x]) = U_s @ U(x) @ U_s^†
    """
    unitaries = {}
    
    for g_idx in range(len(group)):
        V_s = group.get_matrix(g_idx)
        
        if method == 'optimization':
            U_s = solve_for_induced_unitary_optimization(
                X, V_s, encoding_unitary, dim=2**n_qubits
            )
        elif method == 'procrustes':
            U_s = solve_for_induced_unitary_procrustes(
                X, V_s, encoding_unitary, dim=2**n_qubits
            )
        
        unitaries[g_idx] = U_s
    
    return unitaries


import numpy as np
from numpy.linalg import svd, norm

def nearest_unitary(M):
    U, _, Vh = svd(M)
    return U @ Vh

def solve_for_induced_unitary_procrustes(
    data_points: np.ndarray,
    V_s: np.ndarray,
    encoding_unitary: callable,
    dim: int,
    rtol: float = 1e-9,
    verbose: bool = True
):
    """
    Solve (or diagnose) U_s such that U_s U(x) = U(V_s x) U_s
    Uses the *correct* vectorization:
        ( U(x).T ⊗ I  -  I ⊗ U(V_s x) ) vec(U_s) = 0
    
    Returns (U_s_unitary, diagnostics_dict)
    diagnostics_dict contains singular_values, nullspace_dim, residuals, etc.
    """
    data_points = np.atleast_2d(data_points)
    A_blocks = []
    Ux_list = []
    Uvsx_list = []
    for x in data_points:
        Ux = encoding_unitary(x)
        Uvsx = encoding_unitary(V_s @ x)
        Ux_list.append(Ux)
        Uvsx_list.append(Uvsx)

        # CORRECT kron: (Ux.T ⊗ I) - (I ⊗ Uvsx)
        A = np.kron(Ux.T, np.eye(dim)) - np.kron(np.eye(dim), Uvsx)
        A_blocks.append(A)

    A_full = np.vstack(A_blocks)  # shape: (n_points * dim^2, dim^2)

    # SVD
    U_sv, S, Vh = svd(A_full, full_matrices=False)

    # tolerance for zero singular values
    tol = max(A_full.shape) * np.finfo(float).eps * S[0]
    # optionally use user-specified relative tolerance
    tol = max(tol, rtol * S[0])

    null_dim = int(np.sum(S <= tol))

    diagnostics = {
        "singular_values": S,
        "tol": tol,
        "nullspace_dim_estimate": null_dim,
    }

    if verbose:
        print("sv (first few):", S[:min(10, len(S))])
        print("tolerance:", tol, "estimated nullspace dim:", null_dim)

    if null_dim == 0:
        if verbose:
            print("No (numerically) exact nullspace found. The smallest singular value is not ~0.")
        # still take the smallest singular vector (least-squares) but warn it's not exact
        vec = Vh[-1, :]
        M = vec.reshape((dim, dim), order='F')  # column-major reshape (vec stacks columns)
        U_s_candidate = nearest_unitary(M)
        status = "least_squares_candidate_no_exact_nullspace"
    else:
        # If there is a nullspace, get the basis (right-sing vectors corresponding to nullspace)
        null_basis = Vh[-null_dim:, :].conj().T  # shape (dim^2, null_dim)
        # heuristic: try each basis vector as a candidate (after projecting to unitary) and pick the one
        # with smallest residual on the dataset. (This is crude but often useful when null_dim>1.)
        best_res = np.inf
        best_U = None
        for k in range(null_basis.shape[1]):
            veck = null_basis[:, k]
            Mk = veck.reshape((dim, dim), order='F')
            Uk = nearest_unitary(Mk)
            # compute residual summed over points
            res = 0.0
            for Ux, Uvsx in zip(Ux_list, Uvsx_list):
                res += norm(Uk @ Ux - Uvsx @ Uk)**2
            if res < best_res:
                best_res = res
                best_U = Uk
        U_s_candidate = best_U
        status = "exact_nullspace_found" if best_res < 1e-12 else "nullspace_found_but_residual_nonzero_after_projection"

    # compute diagnostics: residuals for candidate and identity
    res_candidate = 0.0
    res_identity = 0.0
    for Ux, Uvsx in zip(Ux_list, Uvsx_list):
        res_candidate += norm(U_s_candidate @ Ux - Uvsx @ U_s_candidate)**2
        res_identity += norm(np.eye(dim) @ Ux - Uvsx @ np.eye(dim))**2

    diagnostics.update({
        "status": status,
        "residual_candidate_fro2": res_candidate,
        "residual_identity_fro2": res_identity,
        "U_s_candidate": U_s_candidate
    })

    if verbose:
        print("status:", status)
        print("residual (candidate)  ||U_s Ux - U(Vs x) U_s||_F^2  =", res_candidate)
        print("residual (identity)   ||I  Ux - U(Vs x) I||_F^2    =", res_identity)

    return U_s_candidate


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