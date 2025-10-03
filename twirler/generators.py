import pennylane as qml
import numpy as np


def _hadamard_generator():
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    G = (X + Z)/np.sqrt(2)
    theta = np.pi
    return G, theta

def _cz_generator():
    Z = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    Z1 = np.kron(Z, I)
    Z2 = np.kron(I, Z)
    Z1Z2 = np.kron(Z, Z)
    G = (np.kron(I,I) - Z1 - Z2 + Z1Z2)/2
    theta = np.pi
    return G, theta

def _cnot_generator():
    Z = np.array([[1,0],[0,-1]])
    X = np.array([[0,1],[1,0]])
    I = np.eye(2)
    Z1 = np.kron(Z, I)
    X2 = np.kron(I, X)
    Z1X2 = np.kron(Z, X)
    G = (np.kron(I,I) - Z1 - X2 + Z1X2)/2
    theta = np.pi
    return G, theta

MANUAL_FIXED_GATE_GENERATORS = {
    "Hadamard": _hadamard_generator,   # single wire
    "CZ": _cz_generator,               # two wires
    "CNOT": _cnot_generator,           # two wires
    "CX": _cnot_generator,             # alias if PennyLane labels as CX
}

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
                results.append((gen.matrix(), op.wires, op.name, theta))
            except Exception as e:
                print(f"Skipping parametric gate {op.name}: {e}")
            continue
        
        # Non-parametric: check manual library
        fn = MANUAL_FIXED_GATE_GENERATORS.get(op.name)
        if fn is not None:
            try:
                if len(op.wires) == 1:
                    gen_obs, theta = fn()
                else:
                    gen_obs, theta = fn()
                results.append((gen_obs, op.wires, op.name, theta))
            except Exception as e:
                print(f"Manual generator failed for {op.name}: {e}")
        # else ignore silently
    
    return results