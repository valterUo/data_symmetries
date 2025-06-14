import json
import os
from pennylane import numpy as np
import pennylane as qml
from concurrent.futures import ProcessPoolExecutor
from ansatz import Ansatz
from efficient_symmetry_group import EfficientSymmetricGroup

num_qubits = 13

dev = qml.device("default.qubit", wires=num_qubits)

def feature_map(features):
    for i, feature in enumerate(features):
        qml.RX(feature[0], wires=i)
        qml.RZ(feature[1], wires=i)

def process_cycle(args):
    """Process a single cycle - recreate circuit inside to avoid pickling issues."""
    cycle, ansatz_id, depth, params_shape_single_qubit, params_shape_two_qubit = args
    
    # Recreate the circuit inside the worker process
    ansatz = Ansatz(ansatz_id, num_qubits, depth)
    circuit = ansatz.get_circuit()
    
    @qml.qnode(dev)
    def full_circuit(params, features):
        feature_map(features)
        
        if params_shape_two_qubit is None:
            circuit(params)
        else:
            circuit(params[0], params[1])
        
        return qml.expval(qml.prod(*[qml.PauliZ(i) for i in range(num_qubits)]))
    
    rotations = cycle[2]
    features = [(rotations[i], rotations[i + 1]) for i in range(1, len(rotations) + 1, 2)]
    res = []
    
    for _ in range(500):  # Reduced for faster execution
        if params_shape_two_qubit is None:
            params_single = np.random.uniform(-np.pi, np.pi, size=params_shape_single_qubit)
            result = full_circuit(params_single, features)
        else:
            params_single = np.random.uniform(-np.pi, np.pi, size=params_shape_single_qubit)
            params_two = np.random.uniform(-np.pi, np.pi, size=params_shape_two_qubit)
            result = full_circuit([params_single, params_two], features)
        res.append(result)
    
    return np.mean(res)

def run_experiment(depth, ansatz_id):
    """Run experiment for specific depth and ansatz."""
    print(f"Starting experiment: Depth {depth}, Ansatz {ansatz_id}")
    
    # Get parameter shapes for this configuration
    ansatz = Ansatz(ansatz_id, num_qubits, depth)
    params_shape_single_qubit, params_shape_two_qubit = ansatz.get_params_shape()
    
    results = {}
    variances = {}
    n = num_qubits * 2
    sym_group = EfficientSymmetricGroup(n)
    
    # Use fewer k values for faster execution
    start = 1
    end = int(np.ceil(n/2))
    
    for k in range(start, end + 1):
        print(f"  Processing {k} cycles (Depth {depth}, Ansatz {ansatz_id})")
        
        k_cycle_datasets = sym_group.generate_k_cycles_dataset(k=k, num_samples=500)
        
        # Prepare arguments for parallel processing (pass parameters instead of function)
        args_list = [(cycle, ansatz_id, depth, params_shape_single_qubit, params_shape_two_qubit) 
                     for cycle in k_cycle_datasets]
        
        #n_cpus = 50

        # Calculate optimal distribution
        num_cycles = len(args_list)
        total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 50))

        # For quantum circuits, balanced approach often works best
        n_workers = min(num_cycles, total_cpus)  # 25 workers max
        threads_per_worker = total_cpus // n_workers  # 2 threads each

        print(f"Quantum simulation: {n_workers} workers × {threads_per_worker} threads on {total_cpus}")
        print(f"Processing {num_cycles} cycles for k={k}")

        # CRITICAL: Set threading limits for quantum circuit simulation
        os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
        os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
        os.environ['NUMBA_NUM_THREADS'] = str(threads_per_worker)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results[int(k)] = list(executor.map(process_cycle, args_list))
    
    # Calculate variances
    for k, values in results.items():
        variance = np.var(values)
        print(f"  {k} cycles variance: {variance}")
        variances[k] = variance
    
    return variances

def save_results(depth, ansatz_id, variances):
    """Save results to JSON file."""
    filename = f'angle_embedding_depth_{depth}.json'
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    existing_data[str(ansatz_id)] = variances
    
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)
    
    print(f"Results saved for Depth {depth}, Ansatz {ansatz_id}")

if __name__ == '__main__':
    # Define experiment parameters
    depths = range(4, 5)      # Depths 1-4
    ansatz_ids = range(7, 20) # Ansatz IDs 1-19
    
    total_experiments = len(depths) * len(ansatz_ids)
    current_experiment = 0
    
    print(f"Starting {total_experiments} experiments...")
    print("="*60)
    
    for depth in depths:
        for ansatz_id in ansatz_ids:
            current_experiment += 1
            print(f"\nExperiment {current_experiment}/{total_experiments}")
            print(f"Depth: {depth}, Ansatz ID: {ansatz_id}")
            print("-" * 40)

            # Run experiment
            variances = run_experiment(depth, ansatz_id)
            
            # Save results
            save_results(depth, ansatz_id, variances)
            
            print(f"✓ Completed: Depth {depth}, Ansatz {ansatz_id}")
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved in files: angle_embedding_depth_1.json to angle_embedding_depth_4.json")
