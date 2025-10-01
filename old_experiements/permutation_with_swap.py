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

dev = qml.device("lightning.qubit", wires=num_qubits)

def feature_map(features):
    for i, feature in enumerate(features):
        qml.RX(feature[0], wires=i)
        qml.RZ(feature[1], wires=i)
        
def insert_swaps(orbits):
    for orbit in orbits:
        # Loop over pairs of qubits
        for i in range(len(orbit) - 1):
            qml.SWAP(wires=[orbit[i] - 1, orbit[i+1] - 1])
        

def process_cycle(cycle):
    orbits = cycle[1]
    features = 2*np.pi*np.random.rand(num_qubits, 2)
    res = []
    
    @qml.qnode(dev)
    def full_circuit(params, features):
        
        feature_map(features)
        insert_swaps(orbits)
        
        if params_shape_two_qubit is None:
            circuit(params)
        else:
            circuit(params[0], params[1])
            
        return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(num_qubits)]))

    
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
    n = num_qubits
    sym_group = EfficientSymmetricGroup(n)

    for k in range(1, int(np.ceil(n/2))):
        print("Processing", k, "cycles")
        results[k] = []
        k_cycle_datasets = sym_group.generate_k_cycles_dataset(k=k, num_samples=1000)
        n_cpus = os.cpu_count()
        with ProcessPoolExecutor(max_workers=n_cpus - 1) as executor:
            results[k] = list(executor.map(process_cycle, k_cycle_datasets))
    
    for k, values in results.items():
        print(k, "cycles variance is ", np.var(values))
        variances[k] = np.var(values)
    
    file = f'results_depth_{depth}_swap.json'
    if os.path.exists(file):
        with open(file, 'r') as f:
            existing_data = json.load(f)
        existing_data[ansatz_id] = variances
    else:
        existing_data = {ansatz_id: variances}
        
    with open(file, 'w') as f:
        json.dump(existing_data, f, indent=4)