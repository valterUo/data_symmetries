import numpy as np
from typing import Dict, List, Tuple
import pennylane as qml

def apply_twirling_to_generators(
    unitaries: Dict[int, np.ndarray],
    generators: List[Tuple],
    full_n_qubits: int
) -> Dict[int, List[np.ndarray]]:
    """
    Apply Pauli twirling formula (equation 52) to generators.
    
    The twirling formula averages the generator over the group:
    G̃ = (1/|G|) Σ_{s∈G} U_s^† G U_s
    
    This produces twirled generators that are symmetric under the group action.
    
    Parameters:
    -----------
    unitaries : Dict[int, np.ndarray]
        Dictionary mapping group element index to unitary matrix U_s
    generators : List[Tuple]
        List of (generator_observable, wires, gate_name) from get_ansatz_generators()
    group_size : int
        Size of the symmetry group |G|
        
    Returns:
    --------
    Dict[int, List[np.ndarray]]
        Dictionary mapping generator index to list of twirled generator matrices,
        one for each group element
    """
    twirled_generators = {}
    
    for gen_idx, (gen_observable, wires, gate_name, theta, is_generator) in enumerate(generators):
        
        op = qml.Hermitian(gen_observable, wires=wires)
        assert op.is_hermitian, "Generator observable must be Hermitian"
        G_full = qml.matrix(op, wire_order=range(full_n_qubits))

        # Apply twirling: G̃_s = U_s^† @ G @ U_s for each group element
        twirled_gen_list = []
        for s in unitaries:
            U_s = unitaries[s]
            U_s_dag = U_s.conj().T
            
            # Twirl the generator
            G_twirled = U_s @ G_full @ U_s_dag
            twirled_gen_list.append(G_twirled)
        
        if len(unitaries) == 1:
            assert np.allclose(twirled_gen_list[0], G_full), "Single element group should not change generator"

        twirled_generators[gen_idx] = {
            'original': G_full,
            'twirled': twirled_gen_list,
            'averaged': np.mean(twirled_gen_list, axis=0),  # Average over group
            'gate_name': gate_name,
            'wires': wires,
            'theta': theta,
            'observable': gen_observable,
            'is_generator': is_generator
        }
    
    return twirled_generators


def compute_twirling_statistics(twirled_generators: Dict) -> Dict:
    """
    Compute statistics about the twirling process.
    
    Parameters:
    -----------
    twirled_generators : Dict
        Output from apply_twirling_to_generators()
        
    Returns:
    --------
    Dict
        Statistics including variance reduction, symmetry preservation, etc.
    """
    statistics = {}
    
    for gen_idx, gen_data in twirled_generators.items():
        original = gen_data['original']
        averaged = gen_data['averaged']
        twirled_list = gen_data['twirled']
        
        # Compute variance across group elements
        variance = np.var([np.linalg.norm(g - averaged, 'fro') for g in twirled_list])
        
        # Distance from original
        distance_from_original = np.linalg.norm(averaged - original, 'fro')
        
        # Check if averaged generator is more symmetric (closer to diagonal/block-diagonal)
        off_diagonal_ratio_original = _compute_off_diagonal_ratio(original)
        off_diagonal_ratio_averaged = _compute_off_diagonal_ratio(averaged)
        
        statistics[gen_idx] = {
            'gate_name': gen_data['gate_name'],
            'variance_across_group': variance,
            'distance_from_original': distance_from_original,
            'off_diagonal_ratio_original': off_diagonal_ratio_original,
            'off_diagonal_ratio_averaged': off_diagonal_ratio_averaged,
            'symmetrization_improvement': off_diagonal_ratio_original - off_diagonal_ratio_averaged
        }
    
    return statistics


def _compute_off_diagonal_ratio(matrix: np.ndarray) -> float:
    """Compute ratio of off-diagonal to total norm."""
    diagonal = np.diag(np.diag(matrix))
    off_diagonal = matrix - diagonal
    return np.linalg.norm(off_diagonal, 'fro') / np.linalg.norm(matrix, 'fro')


def visualize_twirling_effect(twirled_generators: Dict, gen_idx: int = 0):
    """
    Visualize the effect of twirling on a specific generator.
    
    Parameters:
    -----------
    twirled_generators : Dict
        Output from apply_twirling_to_generators()
    gen_idx : int
        Index of generator to visualize
    """
    import matplotlib.pyplot as plt
    
    gen_data = twirled_generators[gen_idx]
    original = gen_data['original']
    averaged = gen_data['averaged']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original generator
    im0 = axes[0].imshow(np.abs(original), cmap='viridis')
    axes[0].set_title(f"Original {gen_data['gate_name']}")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im0, ax=axes[0])
    
    # Averaged (twirled) generator
    im1 = axes[1].imshow(np.abs(averaged), cmap='viridis')
    axes[1].set_title(f"Twirled (Averaged) {gen_data['gate_name']}")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    diff = np.abs(averaged - original)
    im2 = axes[2].imshow(diff, cmap='Reds')
    axes[2].set_title("Absolute Difference")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    print(f"Generator {gen_idx} ({gen_data['gate_name']}) on wires {gen_data['wires']}:")
    print(f"  Frobenius norm change: {np.linalg.norm(diff, 'fro'):.6f}")