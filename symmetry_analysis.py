from typing import Dict

import numpy as np
import scipy


def compute_generator_comparison_metrics(twirled_generators: Dict) -> Dict:
    """
    Compute various metrics comparing original and twirled generators.
    """
    metrics = {}
    
    for gen_idx, elem in twirled_generators.items():
        G_original = elem['original']
        G_twirled = elem['averaged']
        
        # === Distance Metrics ===
        
        # Frobenius norm distance
        frobenius_distance = np.linalg.norm(G_twirled - G_original, 'fro')
        
        # Operator norm distance (largest singular value)
        operator_distance = np.linalg.norm(G_twirled - G_original, 2)
        
        # Trace distance (for normalized operators)
        trace_distance = 0.5 * np.trace(
            scipy.linalg.sqrtm((G_twirled - G_original).conj().T @ (G_twirled - G_original))
        ).real
        
        # Relative distance (normalized by original norm)
        relative_distance = frobenius_distance / np.linalg.norm(G_original, 'fro')
        
        metrics[gen_idx] = {
            'gate_name': elem['gate_name'],
            'wires': elem['wires'],
            'frobenius_distance': frobenius_distance,
            'operator_distance': operator_distance,
            'trace_distance': trace_distance,
            'relative_distance': relative_distance,
        }
    
    return metrics

def compute_symmetry_metrics(G, unitaries, group_size, tolerance=1e-10, print_violations=False) -> Dict:
    """
    Measure how symmetric a generator is under the group action.
    
    A generator is symmetric if: U_s^† G U_s = G for all s in group
    """
    # Measure deviation from perfect symmetry
    symmetry_violations = []
    
    for s in range(group_size):
        U_s = unitaries[s]
        commutator = G @ U_s - U_s @ G
        violation = np.linalg.norm(commutator, 'fro')
        if violation > tolerance and print_violations:
            print(f"Symmetry violation for group element {s}: {violation:.2e}")
            print(f"Commutator", commutator)
        symmetry_violations.append(violation)
    
    return {
        'max_symmetry_violation': np.max(symmetry_violations),
        'mean_symmetry_violation': np.mean(symmetry_violations),
        'is_symmetric': np.max(symmetry_violations) < tolerance
    }


def compare_symmetry_preservation(twirled_generators: Dict, unitaries: Dict, 
                                 group_size: int) -> Dict:
    """
    Compare symmetry properties of original vs twirled generators.
    """
    comparison = {}
    
    for gen_idx, elem in twirled_generators.items():
        G_original = elem['original']
        G_twirled = elem['averaged']
        
        # Measure symmetry of original generator
        sym_original = compute_symmetry_metrics(G_original, unitaries, group_size, print_violations=False)
        
        # Measure symmetry of twirled generator (should be nearly perfect)
        sym_twirled = compute_symmetry_metrics(G_twirled, unitaries, group_size, print_violations=True)
        
        comparison[gen_idx] = {
            'gate_name': elem['gate_name'],
            'original_symmetry': sym_original,
            'twirled_symmetry': sym_twirled,
            'symmetry_improvement': (
                sym_original['max_symmetry_violation'] - 
                sym_twirled['max_symmetry_violation']
            )
        }
    
    return comparison

def compute_spectral_metrics(twirled_generators: Dict) -> Dict:
    """
    Compare eigenvalue structure of original vs twirled generators.
    """
    spectral_metrics = {}
    
    for gen_idx, elem in twirled_generators.items():
        G_original = elem['original']
        G_twirled = elem['averaged']
        
        # Compute eigenvalues
        eigvals_original = np.linalg.eigvals(G_original)
        eigvals_twirled = np.linalg.eigvals(G_twirled)
        
        # Sort eigenvalues by real part
        eigvals_original = np.sort_complex(eigvals_original)
        eigvals_twirled = np.sort_complex(eigvals_twirled)
        
        # Spectral distance (difference in eigenvalues)
        spectral_distance = np.linalg.norm(eigvals_twirled - eigvals_original)
        
        # Entropy of eigenvalue distribution (measure of degeneracy)
        # Normalized eigenvalues as probabilities
        eigvals_orig_normalized = np.abs(eigvals_original) / np.sum(np.abs(eigvals_original))
        eigvals_twirl_normalized = np.abs(eigvals_twirled) / np.sum(np.abs(eigvals_twirled))
        
        entropy_original = -np.sum(eigvals_orig_normalized * np.log(eigvals_orig_normalized + 1e-16))
        entropy_twirled = -np.sum(eigvals_twirl_normalized * np.log(eigvals_twirl_normalized + 1e-16))
        
        # Degeneracy: number of distinct eigenvalues
        degeneracy_original = len(np.unique(np.round(eigvals_original, 10)))
        degeneracy_twirled = len(np.unique(np.round(eigvals_twirled, 10)))
        
        spectral_metrics[gen_idx] = {
            'gate_name': elem['gate_name'],
            'spectral_distance': spectral_distance,
            'entropy_original': entropy_original,
            'entropy_twirled': entropy_twirled,
            'entropy_increase': entropy_twirled - entropy_original,
            'degeneracy_original': degeneracy_original,
            'degeneracy_twirled': degeneracy_twirled,
            'eigenvalues_original': eigvals_original,
            'eigenvalues_twirled': eigvals_twirled
        }
    
    return spectral_metrics

def compute_structure_metrics(G, n_qubits):
    """
    Measure how block-diagonal or structured a generator is.
    Higher symmetry often leads to more block-diagonal structure.
    """
    dim = 2**n_qubits
    
    # Off-diagonal ratio
    diagonal = np.diag(np.diag(G))
    off_diagonal = G - diagonal
    off_diag_ratio = np.linalg.norm(off_diagonal, 'fro') / np.linalg.norm(G, 'fro')
    
    # Block structure: divide into 2x2 blocks and measure coupling
    block_size = 2
    n_blocks = dim // block_size
    inter_block_norm = 0
    intra_block_norm = 0
    
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = G[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if i == j:
                intra_block_norm += np.linalg.norm(block, 'fro')**2
            else:
                inter_block_norm += np.linalg.norm(block, 'fro')**2
    
    block_diagonality = intra_block_norm / (intra_block_norm + inter_block_norm + 1e-16)
    
    # Sparsity (fraction of near-zero entries)
    sparsity = np.sum(np.abs(G) < 1e-10) / G.size
    
    return {
        'off_diagonal_ratio': off_diag_ratio,
        'block_diagonality': block_diagonality,
        'sparsity': sparsity
    }


def compare_structure(twirled_generators: Dict, n_qubits: int) -> Dict:
    """Compare structural properties."""
    comparison = {}
    
    for gen_idx, elem in twirled_generators.items():
        G_original = elem['original']
        G_twirled = elem['averaged']
        
        struct_original = compute_structure_metrics(G_original, n_qubits)
        struct_twirled = compute_structure_metrics(G_twirled, n_qubits)
        
        comparison[gen_idx] = {
            'gate_name': elem['gate_name'],
            'original_structure': struct_original,
            'twirled_structure': struct_twirled,
            'structure_improvement': {
                'block_diagonality': struct_twirled['block_diagonality'] - struct_original['block_diagonality'],
                'sparsity': struct_twirled['sparsity'] - struct_original['sparsity']
            }
        }
    
    return comparison

def compute_commutator_metrics(twirled_generators: Dict, unitaries: Dict, 
                              group_size: int) -> Dict:
    """
    Measure how much generators commute with symmetry operators.
    Symmetric generators should commute with all U_s.
    """
    commutator_metrics = {}
    
    for gen_idx, elem in twirled_generators.items():
        G_original = elem['original']
        G_twirled = elem['averaged']
        
        # Compute commutators [G, U_s] for all group elements
        commutators_original = []
        commutators_twirled = []
        
        for s in range(group_size):
            U_s = unitaries[s]
            
            # Commutator: [G, U_s] = G U_s - U_s G
            comm_orig = G_original @ U_s - U_s @ G_original
            comm_twirl = G_twirled @ U_s - U_s @ G_twirled
            
            commutators_original.append(np.linalg.norm(comm_orig, 'fro'))
            commutators_twirled.append(np.linalg.norm(comm_twirl, 'fro'))
        
        commutator_metrics[gen_idx] = {
            'gate_name': elem['gate_name'],
            'max_commutator_original': np.max(commutators_original),
            'mean_commutator_original': np.mean(commutators_original),
            'max_commutator_twirled': np.max(commutators_twirled),
            'mean_commutator_twirled': np.mean(commutators_twirled),
            'commutator_reduction': np.mean(commutators_original) - np.mean(commutators_twirled)
        }
    
    return commutator_metrics

def comprehensive_generator_comparison(twirled_generators: Dict, unitaries: Dict,
                                      group_size: int, n_qubits: int) -> Dict:
    """
    Complete comparison of original vs twirled generators.
    """
    print("=" * 80)
    print("COMPREHENSIVE GENERATOR COMPARISON")
    print("=" * 80)
    
    # Compute all metrics
    distance_metrics = compute_generator_comparison_metrics(twirled_generators)
    symmetry_comparison = compare_symmetry_preservation(twirled_generators, unitaries, group_size)
    spectral_metrics = compute_spectral_metrics(twirled_generators)
    structure_comparison = compare_structure(twirled_generators, n_qubits)
    commutator_metrics = compute_commutator_metrics(twirled_generators, unitaries, group_size)
    
    # Combine results
    full_comparison = {}
    
    for gen_idx in twirled_generators.keys():
        gate_name = twirled_generators[gen_idx]['gate_name']
        
        print(f"\n{'='*80}")
        print(f"Generator {gen_idx}: {gate_name} on wires {twirled_generators[gen_idx]['wires']}")
        print(f"{'='*80}")
        
        print("\n📏 DISTANCE METRICS:")
        print(f"  Frobenius distance:  {distance_metrics[gen_idx]['frobenius_distance']:.6f}")
        print(f"  Relative distance:   {distance_metrics[gen_idx]['relative_distance']:.6f}")
        print(f"  Operator norm dist:  {distance_metrics[gen_idx]['operator_distance']:.6f}")
        
        print("\n🔄 SYMMETRY METRICS:")
        sym_orig = symmetry_comparison[gen_idx]['original_symmetry']
        sym_twirl = symmetry_comparison[gen_idx]['twirled_symmetry']
        print(f"  Original max violation:  {sym_orig['max_symmetry_violation']:.2e}")
        print(f"  Twirled max violation:   {sym_twirl['max_symmetry_violation']:.2e}")
        print(f"  Symmetry improvement:    {symmetry_comparison[gen_idx]['symmetry_improvement']:.2e}")
        print(f"  Twirled is symmetric:    {sym_twirl['is_symmetric']}")
        
        print("\n📊 SPECTRAL ANALYSIS:")
        print(f"  Spectral distance:       {spectral_metrics[gen_idx]['spectral_distance']:.6f}")
        print(f"  Entropy change:          {spectral_metrics[gen_idx]['entropy_increase']:.6f}")
        print(f"  Degeneracy (original):   {spectral_metrics[gen_idx]['degeneracy_original']}")
        print(f"  Degeneracy (twirled):    {spectral_metrics[gen_idx]['degeneracy_twirled']}")
        
        print("\n🔲 STRUCTURE METRICS:")
        struct_orig = structure_comparison[gen_idx]['original_structure']
        struct_twirl = structure_comparison[gen_idx]['twirled_structure']
        print(f"  Off-diagonal (original): {struct_orig['off_diagonal_ratio']:.4f}")
        print(f"  Off-diagonal (twirled):  {struct_twirl['off_diagonal_ratio']:.4f}")
        print(f"  Block diag. improvement: {structure_comparison[gen_idx]['structure_improvement']['block_diagonality']:.4f}")
        print(f"  Sparsity improvement:    {structure_comparison[gen_idx]['structure_improvement']['sparsity']:.4f}")
        
        print("\n⚡ COMMUTATOR ANALYSIS:")
        print(f"  Mean commutator (orig):  {commutator_metrics[gen_idx]['mean_commutator_original']:.2e}")
        print(f"  Mean commutator (twirl): {commutator_metrics[gen_idx]['mean_commutator_twirled']:.2e}")
        print(f"  Commutator reduction:    {commutator_metrics[gen_idx]['commutator_reduction']:.2e}")
        
        full_comparison[gen_idx] = {
            'distances': distance_metrics[gen_idx],
            'symmetry': symmetry_comparison[gen_idx],
            'spectral': spectral_metrics[gen_idx],
            'structure': structure_comparison[gen_idx],
            'commutators': commutator_metrics[gen_idx]
        }
    
    return full_comparison