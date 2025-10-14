import json
import argparse

import numpy as np
from ansatz import Ansatz
from utils import find_original_param, get_subgroup_unitaries, pennylane_to_qiskit

from qleet.analyzers.entanglement import EntanglementCapability
from qleet.interface.circuit import CircuitDescriptor

from twirler.symmetry_groups import create_symmetric_group
from twirler.generators import get_ansatz_generators
from twirler.twirling import apply_twirling_to_generators

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp

def main(depth, ansatz_id, n_qubits=4):
    final_results = {}
    final_results[n_qubits] = {}

    S = create_symmetric_group(n_qubits)

    with open(f"groups/subgroups_{n_qubits}.json", "r") as f:
        subgroups = json.load(f)

    subgroup_unitaries = get_subgroup_unitaries(subgroups, S)

    print("Depth:", depth)
    final_results[n_qubits][depth] = {}
    print("  Ansatz ID:", ansatz_id)
    final_results[n_qubits][depth][ansatz_id] = {"original": None, "twirled": {}}
    super_ansatz = Ansatz(ansatz_id, n_qubits, depth)
    circuit, params = super_ansatz.get_QNode()
    qiskit_circuit = pennylane_to_qiskit(circuit, n_qubits, params=params)
    qiskit_circuit = qiskit_circuit.remove_final_measurements(inplace=False)
    #qiskit_circuit.draw("mpl", filename=f"original_circuit_n{n_qubits}_d{depth}_a{ansatz_id}.png")
    params = qiskit_circuit.parameters
    circuit_descriptor = CircuitDescriptor(qiskit_circuit, params)
    exp = EntanglementCapability(circuit_descriptor, samples=10000)
    original_entanglement = exp.entanglement_capability()
    print(f"    Original entanglement: {original_entanglement}")
    final_results[n_qubits][depth][ansatz_id]["original"] = original_entanglement

    ansatz_generators = get_ansatz_generators(super_ansatz.get_ansatz())
    for k in subgroup_unitaries:
        final_results[n_qubits][depth][ansatz_id]["twirled"][k] = []
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
                    evo_gate = PauliEvolutionGate(pauli_op, time=param)
                    twirled_circuit.append(evo_gate, range(n_qubits))
                else:
                    raise ValueError(f"Twirled generator for {op_name} on wires {op_wires} not found when {twirled_elem['gate_name']} and {twirled_elem['wires']}")

            #twirled_circuit.remove_final_measurements(inplace=True)
            #transpiled_circuit = transpile(
            #        twirled_circuit,
            #        basis_gates=['rz', 'sx', 'cx', 'h', 'x', 'y', 'z', 'rx', 'ry', 'cz', 'rxx', 'rzz'],
            #        optimization_level=3
            #    )
            #transpiled_circuit.draw("mpl", filename=f"twirled_circuit_n{n_qubits}_d{depth}_a{ansatz_id}_k{k}.png")
            
            params = twirled_circuit.parameters
            assert len(params) == len(qiskit_circuit.parameters), "Parameter count mismatch after twirling."
            circuit_descriptor = CircuitDescriptor(twirled_circuit, params)
            exp = EntanglementCapability(circuit_descriptor, samples=10000)
            twirled_entanglement = exp.entanglement_capability()
            print(f"    Twirled entanglement (k={k}): {twirled_entanglement}")
            final_results[n_qubits][depth][ansatz_id]["twirled"][k].append(twirled_entanglement)

    with open(f"entanglement_results_{n_qubits}_d{depth}_a{ansatz_id}.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entanglement capability experiment for given depth and ansatz_id.")
    parser.add_argument("--depth", type=int, required=True, help="Circuit depth")
    parser.add_argument("--ansatz_id", type=int, required=True, help="Ansatz ID")
    args = parser.parse_args()
    main(args.depth, args.ansatz_id)