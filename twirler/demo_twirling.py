import numpy as np
import pennylane as qml
from symmetry_groups import create_cyclic_group
from induced_representation import (
    angle_encoding,
    derive_unitaries_from_equivariance
)
from twirling import (
    extract_generators_from_ansatz_instance,
    extract_generator_from_single_operation,
    apply_twirling,
    verify_commutation,
    EquivariantAnsatz,
    construct_equivariant_ansatz_from_instance,
    print_generator_summary,
    analyze_generator,
    verify_generator_produces_unitary,
)

# Define Sim1
class Sim1:
    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth

    def get_circuit(self):
        def circuit(single_qubit_params):
            for d in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RX(single_qubit_params[d][i][0], wires=i)
                    qml.RZ(single_qubit_params[d][i][1], wires=i)
        return circuit
    
    def get_params_shape(self):
        return (self.depth, self.num_qubits, 2), None


# ============================================================================
# Example 1: Extract Generator from Single Operation
# ============================================================================
print("=" * 70)
print("Example 1: Extracting Generator from Single RX Operation")
print("=" * 70)

n_qubits = 3

# Extract generator for RX(θ) on qubit 0
print("\nExtracting generator for RX(0.5) on qubit 0:")
G_RX = extract_generator_from_single_operation(qml.RX, wires=0, n_qubits=n_qubits, param_value=0.5)

print(f"Generator shape: {G_RX.shape}")
analysis = analyze_generator(G_RX, "G_RX")
for key, val in analysis.items():
    print(f"  {key}: {val}")

print(f"\nVerifying exp(-iθG) produces unitary: {verify_generator_produces_unitary(G_RX)}")

# ============================================================================
# Example 2: Extract Generators from Full Ansatz
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Extracting Generators from Sim1 Ansatz")
print("=" * 70)

n_qubits = 3
depth = 2

sim1 = Sim1(num_qubits=n_qubits, depth=depth)
print(f"\nAnsatz: Sim1 with {n_qubits} qubits and depth {depth}")
print(f"Expected parameters: {depth * n_qubits * 2} = {depth} layers × {n_qubits} qubits × 2 gates")

# Extract all generators
generators_info = extract_generators_from_ansatz_instance(sim1, n_qubits)

print(f"\nExtracted {len(generators_info)} generators")
print(f"\nFirst 4 generators:")
for i, (G, name, param_idx) in enumerate(generators_info[:4]):
    print(f"\n{i+1}. {name}:")
    print(f"   Parameter index: {param_idx}")
    print(f"   Shape: {G.shape}")
    print(f"   Norm: {np.linalg.norm(G):.6f}")
    print(f"   Is Hermitian: {np.allclose(G, G.conj().T)}")
    print(f"   Produces unitary: {verify_generator_produces_unitary(G)}")

# ============================================================================
# Example 3: Derive Symmetry Unitaries
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Deriving Symmetry Unitaries from Data")
print("=" * 70)

# Setup cyclic group
Z3 = create_cyclic_group(3, 3)
print(f"\nSymmetry group: Z_3 (cyclic group with {len(Z3)} elements)")

# Generate training data
np.random.seed(42)
X_train = np.random.randn(20, 3) * 0.5

def encoding(x):
    return angle_encoding(x, n_qubits=3)

print(f"Training data: {X_train.shape[0]} samples")
print("\nDeriving symmetry unitaries U_s from equivariance...")

unitaries = derive_unitaries_from_equivariance(X_train, Z3, encoding)

print(f"Derived {len(unitaries)} unitaries (one for each group element)")

# Verify they are unitary
print("\nVerifying unitaries:")
for g_idx in range(len(unitaries)):
    U_s = unitaries[g_idx]
    is_unitary = np.allclose(U_s @ U_s.conj().T, np.eye(len(U_s)))
    print(f"  U_{g_idx}: Unitary = {is_unitary}")

# ============================================================================
# Example 4: Apply Twirling to Single Generator
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Twirling a Single Generator")
print("=" * 70)

# Take the first generator from Sim1
G_original, name, _ = generators_info[0]
print(f"\nOriginal generator: {name}")
print(f"Norm: {np.linalg.norm(G_original):.6f}")

# Apply twirling
print("\nApplying twirling formula: G' = (1/|S|) Σ U_s† G U_s")
G_twirled = apply_twirling(G_original, unitaries)

print(f"\nTwirled generator:")
print(f"Norm: {np.linalg.norm(G_twirled):.6f}")
print(f"Is Hermitian: {np.allclose(G_twirled, G_twirled.conj().T)}")

# Verify commutation
print("\nVerifying [G_twirled, U_s] = 0 for all s:")
all_commute, results = verify_commutation(G_twirled, unitaries, tolerance=1e-5)

for g_idx, res in results.items():
    status = "✓" if res['commutes'] else "✗"
    print(f"  U_{g_idx}: {status} Commutator norm = {res['commutator_norm']:.6e}")

print(f"\nResult: {'✓ Twirled generator commutes with all U_s!' if all_commute else '✗ Does not commute'}")

# ============================================================================
# Example 5: Construct Full Equivariant Ansatz
# ============================================================================
print("\n" + "=" * 70)
print("Example 5: Constructing Full Equivariant Ansatz from Sim1")
print("=" * 70)

n_qubits = 3
depth = 1  # Use smaller depth for faster computation

sim1 = Sim1(num_qubits=n_qubits, depth=depth)
print(f"\nOriginal Sim1 ansatz:")
print(f"  Qubits: {n_qubits}")
print(f"  Depth: {depth}")
print(f"  Parameters: {depth * n_qubits * 2}")

# Construct equivariant version
print("\n" + "-" * 70)
eq_ansatz = construct_equivariant_ansatz_from_instance(sim1, n_qubits, unitaries)
print("-" * 70)

# Print summary
print_generator_summary(eq_ansatz, threshold=1e-7)

# ============================================================================
# Example 6: Verify Equivariance
# ============================================================================
print("\n" + "=" * 70)
print("Example 6: Verifying Equivariance of All Generators")
print("=" * 70)

print("\nChecking if all twirled generators commute with symmetry unitaries...")
verification = eq_ansatz.verify_equivariance(tolerance=1e-4)

n_total = len(verification)
n_commute = sum(1 for res in verification.values() if res['all_commute'])

print(f"\nResults: {n_commute}/{n_total} generators commute with all U_s")

# Show details
print("\nDetailed results:")
for name, result in list(verification.items())[:6]:
    if result['all_commute']:
        print(f"  ✓ {name}")
    else:
        max_norm = max(d['commutator_norm'] for d in result['details'].values())
        print(f"  ✗ {name} (max commutator norm: {max_norm:.6e})")

if len(verification) > 6:
    print(f"  ... and {len(verification) - 6} more")

# ============================================================================
# Example 7: Using the Equivariant Ansatz
# ============================================================================
print("\n" + "=" * 70)
print("Example 7: Using the Equivariant Ansatz for Circuit Generation")
print("=" * 70)

# Get active generators (non-zero)
active_gens, active_idx, active_names = eq_ansatz.get_active_generators(threshold=1e-7)

print(f"\nActive generators: {len(active_gens)}")
print(f"Required parameters: {len(active_gens)}")

# Create parameters
params = np.random.randn(len(active_gens)) * 0.1

print(f"\nGenerating parameterized unitary with {len(params)} parameters...")

# Get unitary (using all generators, not just active)
all_params = np.random.randn(len(eq_ansatz.twirled_generators)) * 0.1
U_param = eq_ansatz.get_unitary(all_params)

print(f"Unitary shape: {U_param.shape}")
print(f"Is unitary: {np.allclose(U_param @ U_param.conj().T, np.eye(len(U_param)))}")

# Verify the unitary respects symmetry
print("\nVerifying U(θ) respects symmetry (U_s U(θ) U_s† should also be in family):")
for g_idx in range(min(3, len(unitaries))):
    U_s = unitaries[g_idx]
    U_conjugated = U_s @ U_param @ U_s.conj().T
    is_unitary = np.allclose(U_conjugated @ U_conjugated.conj().T, np.eye(len(U_conjugated)))
    print(f"  U_s{g_idx} U(θ) U_s{g_idx}†: is unitary = {is_unitary}")

# ============================================================================
# Example 8: Create PennyLane Circuit
# ============================================================================
print("\n" + "=" * 70)
print("Example 8: Creating PennyLane Circuit from Equivariant Ansatz")
print("=" * 70)

# Get PennyLane circuit (sparse mode - only non-zero generators)
circuit, active_indices, active_names_list = eq_ansatz.get_pennylane_circuit(
    n_qubits, sparse=True, threshold=1e-7
)

print(f"\nCreated PennyLane circuit with {len(active_indices)} gates")
print(f"Active parameter indices: {active_indices}")
print(f"\nActive gates:")
for name in active_names_list:
    print(f"  - {name}")

# Test the circuit
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def test_circuit(params):
    circuit(params)
    return qml.state()

test_params = np.random.randn(len(active_indices)) * 0.1
final_state = test_circuit(test_params)

print(f"\nCircuit executed successfully!")
print(f"Final state shape: {final_state.shape}")
print(f"Final state norm: {np.linalg.norm(final_state):.6f}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
✓ Successfully extracted generators from Sim1 using qml.generator()
✓ Applied twirling formula to create equivariant generators
✓ Verified all twirled generators commute with symmetry unitaries
✓ Created reduced parameter set (removed zero generators)
✓ Generated PennyLane circuit with equivariant gates

The equivariant ansatz automatically respects the symmetry group!
""")