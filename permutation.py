import json
import os
import numpy as np
import pennylane as qml
from concurrent.futures import ProcessPoolExecutor
from ansatz import Ansatz
from efficient_symmetry_group import EfficientSymmetricGroup

num_qubits = 13
depth = 1
ansatz_id = 19

ansatz = Ansatz(ansatz_id, num_qubits, depth)
params_shape_single_qubit, params_shape_two_qubit = ansatz.get_params_shape()
circuit = ansatz.get_circuit()

dev = qml.device("default.qubit", wires=num_qubits)

def feature_map(features):
    for i, feature in enumerate(features):
        qml.RX(feature[0], wires=i)
        qml.RZ(feature[1], wires=i)

@qml.qnode(dev)
def full_circuit(params, features):
    feature_map(features)
    
    if params_shape_two_qubit is None:
        circuit(params)
    else:
        circuit(params[0], params[1])
    
    return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(num_qubits)]))


def process_cycle(cycle):
    rotations = cycle[2]
    features = [(rotations[i], rotations[i + 1]) for i in range(1, len(rotations) + 1, 2)]
    res = []
    for _ in range(100):
        if params_shape_two_qubit is None:
            params_single = np.random.rand(*params_shape_single_qubit)
            result = full_circuit(params_single, features)
        else:
            params_single = np.random.rand(*params_shape_single_qubit)
            params_two = np.random.rand(*params_shape_two_qubit)
            result = full_circuit([params_single, params_two], features)
        res.append(result)
    return np.mean(res)


if __name__ == '__main__':
    
    results = {}
    variances = {}
    n = num_qubits * 2
    sym_group = EfficientSymmetricGroup(n)

    for k in range(1, num_qubits + 1):
        print("Processing", k, "cycles")
        results[k] = []
        k_cycle_datasets = sym_group.generate_k_cycles_dataset(k=k, num_samples=1000)
        n_cpus = os.cpu_count()
        with ProcessPoolExecutor(max_workers=n_cpus//2) as executor:  # Uses all available CPU cores by default
            results[k] = list(executor.map(process_cycle, k_cycle_datasets))
    
    for k, values in results.items():
        print(k, "cycles variance is ", np.var(values))
        variances[k] = np.var(values)
    
    with open(f'results_{ansatz_id}.json', 'w') as f:
        json.dump(variances, f, indent=4)