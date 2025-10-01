import numpy as np
from symmetry_groups import (
    create_symmetric_group,
    create_cyclic_group,
    permutation_to_matrix,
    check_equivariance,
)

def test_permutation_matrix():
    """Test that permutation matrices work correctly."""
    perm = (1, 0, 2)  # Swap first two elements
    P = permutation_to_matrix(perm, 3)
    
    x = np.array([10, 20, 30])
    result = P @ x
    expected = np.array([20, 10, 30])
    
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ Permutation matrix test passed")


def test_symmetric_group_identity():
    """Test that S_n contains the identity element."""
    S3 = create_symmetric_group(3)
    identity = np.eye(3)
    
    # Find identity in the group
    found_identity = False
    for i in range(len(S3)):
        if np.allclose(S3.get_matrix(i), identity):
            found_identity = True
            break
    
    assert found_identity, "Identity element not found in S_3"
    print("✓ Identity element test passed")


def test_cyclic_group_powers():
    """Test that cyclic group generator satisfies g^n = e."""
    n = 4
    Z4 = create_cyclic_group(n, n)
    
    # Get the generator (first non-identity element)
    g = Z4.get_matrix(1)
    
    # Compute g^n
    g_n = np.linalg.matrix_power(g, n)
    identity = np.eye(n)
    
    assert np.allclose(g_n, identity), "g^n should equal identity"
    print("✓ Cyclic group power test passed")


def test_orbit_size():
    """Test that orbit size equals group order for generic point."""
    S3 = create_symmetric_group(3)
    x = np.array([1.0, 2.0, 3.0])  # Generic point (all different)
    
    from symmetry_groups import compute_orbit
    orbit = compute_orbit(S3, x)
    
    # For generic point, orbit size should equal group order
    assert orbit.shape[0] == len(S3), "Orbit size should equal group order"
    
    # Check that all orbit points are distinct
    unique_rows = np.unique(orbit, axis=0)
    assert len(unique_rows) == len(S3), "All orbit points should be distinct"
    print("✓ Orbit size test passed")


def test_group_closure():
    """Test closure property: V_s @ V_t should be some V_u in the group."""
    S3 = create_symmetric_group(3)
    
    # Pick two elements
    V_s = S3.get_matrix(1)
    V_t = S3.get_matrix(2)
    V_product = V_s @ V_t
    
    # Check if product is in the group
    found = False
    for i in range(len(S3)):
        if np.allclose(V_product, S3.get_matrix(i)):
            found = True
            break
    
    assert found, "Product of group elements should be in the group"
    print("✓ Group closure test passed")


if __name__ == "__main__":
    print("Running tests...\n")
    test_permutation_matrix()
    test_symmetric_group_identity()
    test_cyclic_group_powers()
    test_orbit_size()
    test_group_closure()
    print("\n✅ All tests passed!")