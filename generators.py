import pennylane as qml
import numpy as np

def get_ansatz_generators(ansatz_instance):
    """
    Compute the generators for all gates in any ansatz by recording operations.
    
    Args:
        ansatz_instance: An ansatz object with get_circuit() and get_params_shape() methods
        
    Returns:
        list: List of tuples (generator_observable, wires, gate_name) for each gate
    """
    circuit_func = ansatz_instance.get_circuit()
    params_shape_single, params_shape_two = ansatz_instance.get_params_shape()
    
    # Create dummy parameters with the correct shapes
    if params_shape_two is None:
        dummy_params = np.zeros(params_shape_single)
    else:
        dummy_params_single = np.zeros(params_shape_single)
        dummy_params_two = np.zeros(params_shape_two)
    
    # Create a device to record operations
    dev = qml.device("default.qubit", wires=ansatz_instance.num_qubits)
    
    @qml.qnode(dev)
    def record_circuit():
        if params_shape_two is None:
            circuit_func(dummy_params)
        else:
            circuit_func(dummy_params_single, dummy_params_two)
        return qml.state()
    
    tape = qml.workflow.construct_tape(record_circuit)()

    generators = []
    
    # Extract generators from each operation
    for op in tape.operations:
        try:
            # Get the generator for this operation
            gen = qml.generator(op, format="observable")
            generators.append((gen, op.wires, op.name))
        except Exception as e:
            # Some operations might not have generators (e.g., non-parametric gates)
            print(f"Could not get generator for {op.name}: {e}")
    
    return generators