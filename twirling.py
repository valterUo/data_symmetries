import numpy as np
import pennylane as qml
from typing import Callable, List, Tuple, Optional, Dict
from scipy.linalg import expm

# ============================================================================
# Generator Extraction using PennyLane
# ============================================================================

def extract_generators_from_circuit(
    circuit_func: Callable,
    params: np.ndarray,
    n_qubits: int,
    device_name: str = 'default.qubit'
) -> List[Tuple[np.ndarray, str, int]]:
    """
    Extract all generators from a parameterized circuit by executing it
    and capturing all parametric operations using qml.generator().
    
    Parameters:
    -----------
    circuit_func : Callable
        Circuit function that applies parametric gates
    params : np.ndarray
        Parameters for the circuit (actual values, not dummies)
    n_qubits : int
        Number of qubits
    device_name : str
        PennyLane device name
        
    Returns:
    --------
    List[Tuple[np.ndarray, str, int]]
        List of (generator_matrix, operation_name, param_index) tuples
    """
    dev = qml.device(device_name, wires=n_qubits)
    
    @qml.qnode(dev)
    def qnode_circuit():
        circuit_func(params)
        return qml.state()
    
    # Execute to build tape
    qnode_circuit()
    
    # Access the tape to get operations
    tape = qnode_circuit.qtape
    
    generators = []
    param_idx = 0
    
    for op in tape.operations:
        # Check if operation has parameters (is parametric)
        if op.num_params > 0:
            try:
                # Use qml.generator directly on the operation
                gen_info = qml.generator(op, format='observable')
                
                # Convert to matrix
                G = _observable_to_matrix(gen_info, n_qubits)
                
                # Store generator with metadata
                op_name = f"{op.name}_wires{op.wires.tolist()}_param{param_idx}"
                generators.append((G, op_name, param_idx))
                
                param_idx += op.num_params
                
            except Exception as e:
                print(f"Warning: Could not extract generator for {op.name}: {e}")
                param_idx += op.num_params
                continue
    
    return generators


def _observable_to_matrix(gen_info, n_qubits: int) -> np.ndarray:
    """
    Convert generator info (observable) to matrix.
    
    Parameters:
    -----------
    gen_info : Observable or tuple
        Output from qml.generator()
    n_qubits : int
        Number of qubits
        
    Returns:
    --------
    np.ndarray
        Generator matrix
    """
    if isinstance(gen_info, tuple):
        # Format: (coefficient, observable) or (coefficients, observables)
        if len(gen_info) == 2:
            coeff, obs = gen_info
            if isinstance(obs, list):
                # Multiple observables
                G = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
                for c, ob in zip(coeff, obs):
                    G += c * qml.matrix(ob, wire_order=range(n_qubits))
            else:
                # Single observable with coefficient
                G = coeff * qml.matrix(obs, wire_order=range(n_qubits))
        else:
            raise ValueError(f"Unexpected tuple format: {gen_info}")
    else:
        # Direct observable
        G = qml.matrix(gen_info, wire_order=range(n_qubits))
    
    return G


def extract_generators_from_ansatz_instance(
    ansatz_instance,
    n_qubits: int
) -> List[Tuple[np.ndarray, str, int]]:
    """
    Extract generators from an ansatz instance (like Sim1).
    
    Parameters:
    -----------
    ansatz_instance : object
        Instance of ansatz class with get_circuit() and get_params_shape() methods
    n_qubits : int
        Number of qubits
        
    Returns:
    --------
    List[Tuple[np.ndarray, str, int]]
        List of (generator_matrix, operation_name, param_index) tuples
    """
    circuit_func = ansatz_instance.get_circuit()
    params_shape, _ = ansatz_instance.get_params_shape()
    
    # Create zero parameters of the correct shape
    params = np.ones(params_shape)
    
    return extract_generators_from_circuit(
        circuit_func, params, n_qubits
    )


def extract_generator_from_single_operation(
    op_class,
    wires,
    n_qubits: int,
    param_value: float = 0.0
) -> np.ndarray:
    """
    Extract generator from a single parametric operation.
    
    Parameters:
    -----------
    op_class : pennylane Operation class
        PennyLane operation class (e.g., qml.RX)
    wires : int or list
        Wire(s) the operation acts on
    n_qubits : int
        Total number of qubits
    param_value : float
        Parameter value for the operation
        
    Returns:
    --------
    np.ndarray
        Generator matrix
    """
    # Create the operation instance
    op = op_class(param_value, wires=wires)
    
    # Extract generator
    gen_info = qml.generator(op, format='observable')
    
    # Convert to matrix
    G = _observable_to_matrix(gen_info, n_qubits)
    
    return G


# ============================================================================
# Twirling Formula
# ============================================================================

def apply_twirling(G: np.ndarray, 
                  unitaries: Dict[int, np.ndarray],
                  group_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply the twirling formula to a generator G:
    
    G_twirled = (1/|S|) Σ_{s∈S} U_s^† G U_s
    
    This creates a new generator that commutes with all U_s.
    
    Parameters:
    -----------
    G : np.ndarray
        Original generator (Hermitian matrix)
    unitaries : Dict[int, np.ndarray]
        Dictionary of group unitaries U_s
    group_weights : Optional[np.ndarray]
        Weights for each group element (uniform if None)
        
    Returns:
    --------
    np.ndarray
        Twirled generator G_twirled (Hermitian)
    """
    group_size = len(unitaries)
    
    if group_weights is None:
        group_weights = np.ones(group_size) / group_size
    
    dim = G.shape[0]
    G_twirled = np.zeros((dim, dim), dtype=complex)
    
    for g_idx in range(group_size):
        U_s = unitaries[g_idx]
        # Apply conjugation: U_s^† G U_s
        G_twirled += group_weights[g_idx] * (U_s.conj().T @ G @ U_s)
    
    # Ensure Hermitian
    G_twirled = (G_twirled + G_twirled.conj().T) / 2
    
    return G_twirled


def verify_commutation(G: np.ndarray,
                      unitaries: Dict[int, np.ndarray],
                      tolerance: float = 1e-6) -> Tuple[bool, dict]:
    """
    Verify that generator G commutes with all unitaries U_s:
    [exp(-iθG), U_s] = 0
    
    Equivalently: [G, U_s] = 0
    
    Parameters:
    -----------
    G : np.ndarray
        Generator matrix
    unitaries : Dict[int, np.ndarray]
        Dictionary of unitaries
    tolerance : float
        Numerical tolerance
        
    Returns:
    --------
    Tuple[bool, dict]
        (all_commute, detailed_results)
    """
    results = {}
    all_commute = True
    
    for g_idx, U_s in unitaries.items():
        # Compute commutator [G, U_s] = G U_s - U_s G
        commutator = G @ U_s - U_s @ G
        commutator_norm = np.linalg.norm(commutator)
        
        commutes = commutator_norm < tolerance
        results[g_idx] = {
            'commutator_norm': commutator_norm,
            'commutes': commutes
        }
        
        if not commutes:
            all_commute = False
    
    return all_commute, results


# ============================================================================
# Equivariant Ansatz Construction
# ============================================================================

class EquivariantAnsatz:
    """
    Equivariant ansatz constructed by twirling generators.
    """
    
    def __init__(self,
                 generators: List[np.ndarray],
                 unitaries: Dict[int, np.ndarray],
                 names: Optional[List[str]] = None,
                 param_indices: Optional[List[int]] = None):
        """
        Initialize equivariant ansatz.
        
        Parameters:
        -----------
        generators : List[np.ndarray]
            Original (non-equivariant) generators
        unitaries : Dict[int, np.ndarray]
            Symmetry unitaries U_s
        names : Optional[List[str]]
            Names for each generator
        param_indices : Optional[List[int]]
            Parameter indices for each generator
        """
        self.original_generators = generators
        self.unitaries = unitaries
        self.names = names or [f"G_{i}" for i in range(len(generators))]
        self.param_indices = param_indices or list(range(len(generators)))
        
        # Apply twirling to all generators
        print(f"Twirling {len(generators)} generators...")
        self.twirled_generators = []
        for i, G in enumerate(generators):
            G_twirled = apply_twirling(G, unitaries)
            self.twirled_generators.append(G_twirled)
            if (i + 1) % 10 == 0 or (i + 1) == len(generators):
                print(f"  Progress: {i + 1}/{len(generators)}")
        print(f"Twirling complete!")
    
    def get_unitary(self, params: np.ndarray) -> np.ndarray:
        """
        Get the parameterized unitary: U(θ) = ∏_i exp(-iθ_i G_i^twirled)
        
        Parameters:
        -----------
        params : np.ndarray
            Parameters θ_i for each generator
            
        Returns:
        --------
        np.ndarray
            Parameterized unitary matrix
        """
        if len(params) != len(self.twirled_generators):
            raise ValueError(f"Expected {len(self.twirled_generators)} params, got {len(params)}")
        
        dim = self.twirled_generators[0].shape[0]
        U = np.eye(dim, dtype=complex)
        
        for theta, G in zip(params, self.twirled_generators):
            U = U @ expm(-1j * theta * G)
        
        return U
    
    def verify_equivariance(self, tolerance: float = 1e-6) -> dict:
        """
        Verify that all twirled generators commute with symmetry unitaries.
        
        Returns:
        --------
        dict
            Verification results for each generator
        """
        results = {}
        
        for i, (G, name) in enumerate(zip(self.twirled_generators, self.names)):
            all_commute, details = verify_commutation(G, self.unitaries, tolerance)
            results[name] = {
                'all_commute': all_commute,
                'details': details
            }
        
        return results
    
    def get_active_generators(self, threshold: float = 1e-8) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Get only non-zero generators (for parameter reduction).
        
        Parameters:
        -----------
        threshold : float
            Threshold for considering a generator as zero
            
        Returns:
        --------
        Tuple[List[np.ndarray], List[int], List[str]]
            (active_generators, active_indices, active_names)
        """
        active_generators = []
        active_indices = []
        active_names = []
        
        for i, (G, name) in enumerate(zip(self.twirled_generators, self.names)):
            norm = np.linalg.norm(G)
            if norm > threshold:
                active_generators.append(G)
                active_indices.append(i)
                active_names.append(name)
        
        return active_generators, active_indices, active_names
    
    def get_pennylane_circuit(self, num_qubits: int, sparse: bool = True, threshold: float = 1e-8):
        """
        Create a PennyLane circuit using the twirled generators.
        
        Parameters:
        -----------
        num_qubits : int
            Number of qubits
        sparse : bool
            If True, only include non-zero generators
        threshold : float
            Threshold for sparse mode
            
        Returns:
        --------
        Tuple[Callable, List[int], List[str]]
            (circuit_function, active_indices, active_names)
        """
        if sparse:
            active_generators, active_indices, active_names = self.get_active_generators(threshold)
        else:
            active_generators = self.twirled_generators
            active_indices = list(range(len(self.twirled_generators)))
            active_names = self.names
        
        def circuit(params):
            """PennyLane circuit applying exp(-iθG) for each twirled generator."""
            if len(params) != len(active_generators):
                raise ValueError(f"Expected {len(active_generators)} parameters, got {len(params)}")
            
            for theta, G in zip(params, active_generators):
                # Use qml.QubitUnitary to apply exp(-iθG)
                U_theta = expm(-1j * theta * G)
                qml.QubitUnitary(U_theta, wires=range(num_qubits))
        
        return circuit, active_indices, active_names


# ============================================================================
# Utilities
# ============================================================================

def analyze_generator(G: np.ndarray, name: str = "G") -> dict:
    """
    Analyze properties of a generator matrix.
    
    Parameters:
    -----------
    G : np.ndarray
        Generator matrix
    name : str
        Name of the generator
        
    Returns:
    --------
    dict
        Properties of the generator
    """
    eigenvalues = np.linalg.eigvals(G)
    
    return {
        'name': name,
        'shape': G.shape,
        'is_hermitian': np.allclose(G, G.conj().T),
        'norm': np.linalg.norm(G),
        'trace': np.trace(G),
        'eigenvalues_real': np.allclose(eigenvalues.imag, 0),
        'eigenvalue_range': (np.min(eigenvalues.real), np.max(eigenvalues.real)),
    }


def compare_generators(G_original: np.ndarray,
                      G_twirled: np.ndarray) -> dict:
    """
    Compare original and twirled generators.
    
    Parameters:
    -----------
    G_original : np.ndarray
        Original generator
    G_twirled : np.ndarray
        Twirled generator
        
    Returns:
    --------
    dict
        Comparison metrics
    """
    diff_norm = np.linalg.norm(G_original - G_twirled)
    relative_diff = diff_norm / (np.linalg.norm(G_original) + 1e-10)
    
    return {
        'difference_norm': diff_norm,
        'relative_difference': relative_diff,
        'original_norm': np.linalg.norm(G_original),
        'twirled_norm': np.linalg.norm(G_twirled),
    }


def construct_equivariant_ansatz_from_instance(
    ansatz_instance,
    n_qubits: int,
    unitaries: Dict[int, np.ndarray]
) -> EquivariantAnsatz:
    """
    Construct an equivariant ansatz from an ansatz instance (like Sim1).
    
    Parameters:
    -----------
    ansatz_instance : object
        Ansatz instance with get_circuit() and get_params_shape() methods
    n_qubits : int
        Number of qubits
    unitaries : Dict[int, np.ndarray]
        Symmetry unitaries
        
    Returns:
    --------
    EquivariantAnsatz
        Equivariant version of the ansatz
    """
    # Extract generators
    print(f"Extracting generators from ansatz...")
    generators_with_info = extract_generators_from_ansatz_instance(
        ansatz_instance, n_qubits
    )
    
    generators = [g for g, _, _ in generators_with_info]
    names = [name for _, name, _ in generators_with_info]
    param_indices = [idx for _, _, idx in generators_with_info]
    
    print(f"Extracted {len(generators)} generators")
    
    # Create equivariant ansatz
    eq_ansatz = EquivariantAnsatz(generators, unitaries, names, param_indices)
    
    return eq_ansatz


def verify_generator_produces_unitary(G: np.ndarray, theta: float = 0.5) -> bool:
    """
    Verify that exp(-iθG) is unitary.
    
    Parameters:
    -----------
    G : np.ndarray
        Generator matrix
    theta : float
        Parameter value
        
    Returns:
    --------
    bool
        True if exp(-iθG) is unitary
    """
    U = expm(-1j * theta * G)
    identity = np.eye(len(U))
    return np.allclose(U @ U.conj().T, identity, atol=1e-8)


def print_generator_summary(eq_ansatz: EquivariantAnsatz, threshold: float = 1e-8):
    """
    Print a summary of the equivariant ansatz generators.
    
    Parameters:
    -----------
    eq_ansatz : EquivariantAnsatz
        Equivariant ansatz
    threshold : float
        Threshold for considering generators as zero
    """
    print(f"\n{'='*70}")
    print("Generator Summary")
    print(f"{'='*70}")
    
    print(f"\nTotal generators: {len(eq_ansatz.twirled_generators)}")
    
    active_gens, active_idx, active_names = eq_ansatz.get_active_generators(threshold)
    
    print(f"Active (non-zero) generators: {len(active_gens)}")
    print(f"Inactive (zero) generators: {len(eq_ansatz.twirled_generators) - len(active_gens)}")
    print(f"Parameter reduction: {len(eq_ansatz.twirled_generators) - len(active_gens)} "
          f"({100*(len(eq_ansatz.twirled_generators) - len(active_gens))/len(eq_ansatz.twirled_generators):.1f}%)")
    
    print(f"\nActive generators:")
    for i, (name, G) in enumerate(zip(active_names[:10], active_gens[:10])):
        norm = np.linalg.norm(G)
        print(f"  {name}: norm = {norm:.6f}")
    
    if len(active_gens) > 10:
        print(f"  ... and {len(active_gens) - 10} more")