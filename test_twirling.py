import numpy as np
from scipy.linalg import expm
from symmetry_groups import create_cyclic_group
from induced_representation import angle_encoding, derive_unitaries_from_equivariance
from twirling import (
    extract_generator_from_rotation,
    apply_twirling,
    verify_commutation,
    EquivariantAnsatz,
)


def test_generator_is_hermitian():
    """Test that extracted generators are Hermitian."""
    G = extract_generator_from_rotation('X', qubit=0, n_qubits=2)
    
    assert np.allclose(G, G.conj().T), "Generator should be Hermitian"
    print("✓ Generator is Hermitian test passed")


def test_generator_produces_unitary():
    """Test that exp(-iθG) produces a unitary matrix."""
    G = extract_generator_from_rotation('Z', qubit=1, n_qubits=3)
    
    theta = 0.5
    U = expm(-1j * theta * G)
    
    identity = np.eye(len(U))
    assert np.allclose(U @ U.conj().T, identity), "exp(-iθG) should be unitary"
    print("✓ Generator produces unitary test passed")


def test_twirled_generator_is_hermitian():
    """Test that twirled generators remain Hermitian."""
    # Setup
    Z3 = create_cyclic_group(3, 3)
    X = np.random.randn(10, 3) * 0.5
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    unitaries = derive_unitaries_from_equivariance(X, Z3, encoding)
    
    G = extract_generator_from_rotation('X', qubit=0, n_qubits=3)
    G_twirled = apply_twirling(G, unitaries)
    
    assert np.allclose(G_twirled, G_twirled.conj().T), "Twirled generator should be Hermitian"
    print("✓ Twirled generator is Hermitian test passed")


def test_twirled_generator_commutes():
    """Test that twirled generators commute with all unitaries."""
    # Setup
    Z4 = create_cyclic_group(4, 4)
    X = np.random.randn(12, 4)
    
    def encoding(x):
        return angle_encoding(x, n_qubits=4)
    
    unitaries = derive_unitaries_from_equivariance(X, Z4, encoding)
    
    G = extract_generator_from_rotation('Y', qubit=2, n_qubits=4)
    G_twirled = apply_twirling(G, unitaries)
    
    all_commute, results = verify_commutation(G_twirled, unitaries, tolerance=1e-4)
    
    assert all_commute, "Twirled generator should commute with all unitaries"
    print("✓ Twirled generator commutes test passed")


def test_equivariant_ansatz_construction():
    """Test that EquivariantAnsatz can be constructed."""
    Z3 = create_cyclic_group(3, 3)
    X = np.random.randn(10, 3) * 0.5
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    unitaries = derive_unitaries_from_equivariance(X, Z3, encoding)
    
    generators = [
        extract_generator_from_rotation('X', 0, 3),
        extract_generator_from_rotation('Z', 1, 3),
    ]
    
    eq_ansatz = EquivariantAnsatz(generators, unitaries)
    
    assert len(eq_ansatz.twirled_generators) == len(generators)
    assert len(eq_ansatz.original_generators) == len(generators)
    print("✓ Equivariant ansatz construction test passed")


def test_equivariant_ansatz_produces_unitary():
    """Test that parameterized equivariant ansatz produces unitary."""
    Z3 = create_cyclic_group(3, 3)
    X = np.random.randn(10, 3) * 0.5
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    unitaries = derive_unitaries_from_equivariance(X, Z3, encoding)
    
    generators = [
        extract_generator_from_rotation('X', 0, 3),
        extract_generator_from_rotation('Y', 1, 3),
    ]
    
    eq_ansatz = EquivariantAnsatz(generators, unitaries)
    
    params = np.array([0.3, 0.5])
    U = eq_ansatz.get_unitary(params)
    
    identity = np.eye(len(U))
    assert np.allclose(U @ U.conj().T, identity), "Parameterized unitary should be unitary"
    print("✓ Equivariant ansatz produces unitary test passed")


def test_equivariant_ansatz_verification():
    """Test that equivariant ansatz verification works."""
    Z3 = create_cyclic_group(3, 3)
    X = np.random.randn(10, 3) * 0.5
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    unitaries = derive_unitaries_from_equivariance(X, Z3, encoding)
    
    generators = [extract_generator_from_rotation('Z', 0, 3)]
    
    eq_ansatz = EquivariantAnsatz(generators, unitaries)
    
    verification = eq_ansatz.verify_equivariance(tolerance=1e-4)
    
    for name, result in verification.items():
        assert result['all_commute'], f"Generator {name} should commute with all unitaries"
    
    print("✓ Equivariant ansatz verification test passed")


if __name__ == "__main__":
    print("Running twirling tests...\n")
    np.random.seed(123)
    
    test_generator_is_hermitian()
    test_generator_produces_unitary()
    test_twirled_generator_is_hermitian()
    test_twirled_generator_commutes()
    test_equivariant_ansatz_construction()
    test_equivariant_ansatz_produces_unitary()
    test_equivariant_ansatz_verification()
    
    print("\n✅ All twirling tests passed!")