import numpy as np
from symmetry_groups import create_symmetric_group, create_cyclic_group
from induced_representation import *

def test_angle_encoding_normalization():
    """Test that angle encoding produces normalized states."""
    x = np.array([0.5, 1.0, 1.5])
    state = angle_encoding(x, n_qubits=3)
    
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0), f"State should be normalized, got norm={norm}"
    print("✓ Angle encoding normalization test passed")


def test_amplitude_encoding_normalization():
    """Test that amplitude encoding produces normalized states."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    state = amplitude_encoding(x)
    
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0), f"State should be normalized, got norm={norm}"
    print("✓ Amplitude encoding normalization test passed")


def test_derived_unitaries_are_unitary():
    """Test that all derived unitaries are actually unitary."""
    S3 = create_symmetric_group(3)
    X = np.random.randn(5, 3) * 0.5
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    unitaries = derive_unitaries_from_equivariance(X, S3, encoding)
    
    for g_idx, U_s in unitaries.items():
        assert check_unitary(U_s), f"U_{g_idx} is not unitary"
    
    print("✓ Derived unitaries are unitary test passed")


def test_equivariance_condition():
    """Test that derived unitaries satisfy the equivariance condition."""
    Z4 = create_cyclic_group(4, 4)
    X = np.random.randn(10, 4)
    
    def encoding(x):
        return amplitude_encoding(x)
    
    unitaries = derive_unitaries_from_equivariance(X, Z4, encoding)
    all_passed, results = verify_equivariance(X, Z4, encoding, unitaries, tolerance=1e-3)
    
    assert all_passed, "Equivariance condition not satisfied"
    print("✓ Equivariance condition test passed")


def test_identity_element_maps_to_identity():
    """Test that the identity group element gives identity unitary."""
    S3 = create_symmetric_group(3)
    X = np.random.randn(8, 3) * 0.5
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    unitaries = derive_unitaries_from_equivariance(X, S3, encoding)
    
    # Find identity element (permutation (0,1,2))
    identity_idx = None
    for i, elem in enumerate(S3.elements):
        if elem == (0, 1, 2):
            identity_idx = i
            break
    
    assert identity_idx is not None, "Identity element not found"
    
    U_identity = unitaries[identity_idx]
    state_dim = len(U_identity)
    I = np.eye(state_dim)
    
    # Check if U_identity is close to identity (up to global phase)
    error = min(
        np.linalg.norm(U_identity - I),
        np.linalg.norm(U_identity + I)
    )
    
    assert error < 0.1, f"Identity element should map to identity unitary, error={error}"
    print("✓ Identity element test passed")


def test_group_homomorphism():
    """Test that U_s forms a group homomorphism (approximately)."""
    # For cyclic group Z_3, we should have (U_g)^3 ≈ I
    Z3 = create_cyclic_group(3, 3)
    X = np.random.randn(10, 3)
    
    def encoding(x):
        return amplitude_encoding(x)
    
    unitaries = derive_unitaries_from_equivariance(X, Z3, encoding)
    
    # Get the generator (element 1)
    U_g = unitaries[1]
    
    # Compute (U_g)^3
    U_g_cubed = U_g @ U_g @ U_g
    
    state_dim = len(U_g)
    I = np.eye(state_dim)
    
    # Check if close to identity (up to global phase)
    error = min(
        np.linalg.norm(U_g_cubed - I),
        np.linalg.norm(U_g_cubed + I)
    )
    
    assert error < 0.2, f"(U_g)^3 should be close to identity, error={error}"
    print("✓ Group homomorphism test passed")


def test_consistency_across_datasets():
    """Test that derived unitaries are consistent across different datasets."""
    S3 = create_symmetric_group(3)
    
    def encoding(x):
        return angle_encoding(x, n_qubits=3)
    
    # Two different datasets
    X1 = np.random.randn(10, 3) * 0.5
    X2 = np.random.randn(10, 3) * 0.5
    
    unitaries1 = derive_unitaries_from_equivariance(X1, S3, encoding)
    unitaries2 = derive_unitaries_from_equivariance(X2, S3, encoding)
    
    # Unitaries should be similar (up to numerical errors and global phase)
    for g_idx in range(len(S3)):
        U1 = unitaries1[g_idx]
        U2 = unitaries2[g_idx]
        
        # Check distance (account for global phase)
        error = min(
            np.linalg.norm(U1 - U2),
            np.linalg.norm(U1 + U2)
        )
        
        # Tolerance is higher here since different datasets may give slightly different results
        assert error < 0.5, f"Unitaries for element {g_idx} differ too much: {error}"
    
    print("✓ Consistency across datasets test passed")


if __name__ == "__main__":
    print("Running tests for derived unitaries...\n")
    np.random.seed(42)
    
    test_angle_encoding_normalization()
    test_amplitude_encoding_normalization()
    test_derived_unitaries_are_unitary()
    test_equivariance_condition()
    test_identity_element_maps_to_identity()
    test_group_homomorphism()
    test_consistency_across_datasets()
    
    print("\n✅ All tests passed!")