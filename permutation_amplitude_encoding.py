import json
import os
import numpy as np
import pennylane as qml
from concurrent.futures import ProcessPoolExecutor
from ansatz import Ansatz
from efficient_symmetry_group import EfficientSymmetricGroup

num_qubits = 13
depth = 2
ansatz_id = 6

ansatz = Ansatz(ansatz_id, num_qubits, depth)
params_shape_single_qubit, params_shape_two_qubit = ansatz.get_params_shape()
circuit = ansatz.get_circuit()

dev = qml.device("lightning.qubit", wires=num_qubits)

@qml.qnode(dev)
def full_circuit(params, features):
    qml.AmplitudeEmbedding(features, wires=range(num_qubits), normalize=True)
    
    if params_shape_two_qubit is None:
        circuit(params)
    else:
        circuit(params[0], params[1])
    
    return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(num_qubits)]))


def process_cycle(cycle):
    features = list(cycle[2])
    res = []
    for _ in range(200):
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
    n = 2 ** num_qubits
    sym_group = EfficientSymmetricGroup(n)
    start = 1
    end = int(np.ceil(n/2))
    num_points = 20
    k_values = np.linspace(start, end, num_points, dtype=int)

    for k in k_values:
        print("Processing", k, "cycles")
        k_cycle_datasets = sym_group.generate_k_cycles_dataset(k=k, num_samples=1000)
        n_cpus = os.cpu_count()
        with ProcessPoolExecutor(max_workers=int(n_cpus-2)) as executor:
            results[int(k)] = list(executor.map(process_cycle, k_cycle_datasets))
    
    for k, values in results.items():
        print(k, "cycles variance is ", np.var(values))
        variances[k] = np.var(values)
    
    file = f'amplitude_embedding_results_depth_{depth}.json'
    if os.path.exists(file):
        with open(file, 'r') as f:
            existing_data = json.load(f)
        existing_data[ansatz_id] = variances
    else:
        existing_data = {ansatz_id: variances}
        
    with open(file, 'w') as f:
        json.dump(existing_data, f, indent=4)