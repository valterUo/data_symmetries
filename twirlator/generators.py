import pennylane as qml
import numpy as np
from qiskit.quantum_info import SparsePauliOp

def _hadamard_generator():
    # H = (1/√2) * (X + Z)
    pauli_op = SparsePauliOp.from_list([
        ("X", 1/np.sqrt(2)),
        ("Z", 1/np.sqrt(2)),
    ])
    theta = np.pi / 2
    return pauli_op, theta

def _cz_generator():
    # CZ = exp(-i * pi * ((I - Z)/2 ⊗ (I - Z)/2)) exactly
    #     = exp(-i * pi * (II - IZ - ZI + ZZ)/4)
    pauli_op = SparsePauliOp.from_list([
        ("II",  1/4),
        ("IZ", -1/4),
        ("ZI", -1/4),
        ("ZZ",  1/4),
    ])
    theta = np.pi
    return pauli_op, theta

def _cnot_generator():
    # H = 0.5 * (I⊗I - I⊗X - Z⊗I + Z⊗X)
    pauli_op = SparsePauliOp.from_list([
        ("II",  1/2),
        ("IX", -1/2),
        ("ZI", -1/2),
        ("ZX",  1/2),
    ])
    theta = np.pi / 2
    return pauli_op, theta

def get_ansatz_generators(ansatz_instance):
    """
    Compute generators for parameterized gates; optionally attach
    manual (fixed-angle) generators for selected fixed Clifford gates.
    
    Returns:
        list of tuples: (generator_observable, wires, gate_name, theta_or_None, is_manual)
            - theta_or_None: recommended angle (if we expressed gate as exp(-i θ/2 G))
            - is_manual: bool flag (True if from manual fixed gate mapping)
    """
    circuit_func = ansatz_instance.get_circuit()
    params_shape_single, params_shape_two = ansatz_instance.get_params_shape()
    
    if params_shape_two is None:
        dummy_params = np.zeros(params_shape_single)
    else:
        dummy_params_single = np.zeros(params_shape_single)
        dummy_params_two = np.zeros(params_shape_two)
    
    dev = qml.device("default.qubit", wires=ansatz_instance.num_qubits)

    @qml.qnode(dev)
    def record_circuit():
        if params_shape_two is None:
            circuit_func(dummy_params)
        else:
            circuit_func(dummy_params_single, dummy_params_two)
        return qml.state()
    
    tape = qml.workflow.construct_tape(record_circuit)()
    results = []
    
    for op in tape.operations:
        # Parametric gates: use qml.generator directly
        if op.num_params > 0:
            try:
                gen = qml.generator(op, format="observable")
                # In PennyLane convention U(θ)=exp(-i θ/2 G) with current θ=op.parameters[0]
                theta = op.parameters[0] if op.parameters else None
                results.append((gen.matrix(), op.wires, op.name, theta, True))
            except Exception as e:
                print(f"Skipping parametric gate {op.name}: {e}")
            continue
        
        # Non-parametric gates: return gate's unitary representation
        if op.name == "CNOT":
            G, theta = _cnot_generator()
            results.append((G.to_matrix(), op.wires, op.name, theta, False))
        elif op.name == "CZ":
            G, theta = _cz_generator()
            results.append((G.to_matrix(), op.wires, op.name, theta, False))
        elif op.name == "Hadamard":
            G, theta = _hadamard_generator()
            results.append((G.to_matrix(), op.wires, op.name, theta, False))
        else:
            raise ValueError(f"Generator for fixed gate {op.name} not implemented.")
    
    return results