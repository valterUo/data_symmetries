import json
import argparse
from ansatz import Ansatz
from utils import find_original_param, get_subgroup_unitaries, pennylane_to_qiskit

from qleet.analyzers.expressibility import Expressibility
from qleet.interface.circuit import CircuitDescriptor

from twirler.symmetry_groups import create_symmetric_group
from twirler.generators import get_ansatz_generators
from twirler.twirling import apply_twirling_to_generators

from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate
from qiskit.quantum_info import SparsePauliOp

def main(depth, ansatz_id):
    final_results = {}
    n_qubits = 4

    S = create_symmetric_group(n_qubits)

    with open(f"groups/subgroups_{n_qubits}.json", "r") as f:
        subgroups = json.load(f)

    subgroup_unitaries = get_subgroup_unitaries(subgroups, S)

    print("Depth:", depth)
    final_results[str(depth)] = {}
    print("  Ansatz ID:", ansatz_id)
    final_results[str(depth)][str(ansatz_id)] = {"original": None, "twirled": {}}
    super_ansatz = Ansatz(ansatz_id, n_qubits, depth)
    circuit, params = super_ansatz.get_QNode()
    qiskit_circuit = pennylane_to_qiskit(circuit, n_qubits, params=params)
    qiskit_circuit.remove_final_measurements()
    params = qiskit_circuit.parameters
    circuit_descriptor = CircuitDescriptor(qiskit_circuit, params)
    exp = Expressibility(circuit_descriptor, samples=5000)
    original_expressibility = exp.expressibility()
    print(f"    Original expressibility = {original_expressibility:.4f}")
    final_results[str(depth)][str(ansatz_id)]["original"] = original_expressibility
    ansatz_generators = get_ansatz_generators(super_ansatz.get_ansatz())

    for k in subgroup_unitaries:
        final_results[str(depth)][str(ansatz_id)]["twirled"][str(k)] = []
        for elem in subgroup_unitaries[k]:
            unitaries = elem["unitaries"]
            twirled_generators = apply_twirling_to_generators(unitaries, ansatz_generators, n_qubits)
            twirled_circuit = QuantumCircuit(n_qubits)
            for i, (gen_matrix, op_wires, op_name, theta, is_parametric) in enumerate(ansatz_generators):
                twirled_elem = twirled_generators[i]
                if twirled_elem["gate_name"] == op_name and twirled_elem["wires"] == op_wires:
                    H = twirled_elem['averaged']
                    if is_parametric:
                        param = find_original_param(qiskit_circuit, ansatz_generators, i, op_name, op_wires, theta)
                    else:
                        param = theta
                    pauli_op = SparsePauliOp.from_operator(H)
                    evo_gate = HamiltonianGate(pauli_op, time=param)
                    twirled_circuit.append(evo_gate, range(n_qubits))
                else:
                    raise ValueError(f"Twirled generator for {op_name} on wires {op_wires} not found when {twirled_elem['gate_name']} and {twirled_elem['wires']}")
            
            twirled_circuit.remove_final_measurements(inplace=True)
            params = twirled_circuit.parameters

            assert len(params) == len(qiskit_circuit.parameters), "Parameter count mismatch after twirling"

            circuit_descriptor = CircuitDescriptor(twirled_circuit, params)
            exp = Expressibility(circuit_descriptor, samples=5000)
            twirled_expressibility = exp.expressibility()
            print(f"    Subgroup {k}: expressibility = {twirled_expressibility:.4f}")
            final_results[str(depth)][str(ansatz_id)]["twirled"][str(k)].append(twirled_expressibility)

    with open(f"results/expressibility_results_{n_qubits}_d{depth}_a{ansatz_id}.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute expressibility for given depth and ansatz_id.")
    parser.add_argument("--depth", type=int, required=True, help="Circuit depth")
    parser.add_argument("--ansatz_id", type=int, required=True, help="Ansatz ID")
    args = parser.parse_args()
    main(args.depth, args.ansatz_id)