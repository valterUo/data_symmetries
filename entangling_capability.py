import numpy as np
import json
import argparse
from tqdm import tqdm
from ansatz import Ansatz
from utils import pennylane_to_qiskit
import pennylane as qml

from qleet.analyzers.entanglement import EntanglementCapability
from qleet.interface.circuit import CircuitDescriptor

from twirler.symmetry_groups import create_subgroup_from_permutations, create_symmetric_group
from twirler.induced_representation import derive_unitaries_angle_embedding_analytic
from twirler.generators import get_ansatz_generators
from twirler.twirling import apply_twirling_to_generators

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

def main(depth, ansatz_id, n_qubits=4):
    final_results = {}
    final_results[n_qubits] = {}

    S = create_symmetric_group(n_qubits)

    with open(f"groups/subgroups_{n_qubits}.json", "r") as f:
        subgroups = json.load(f)

    subgroup_unitaries = {}
    for k, groups in subgroups.items():
        subgroup_unitaries[k] = []
        for generators in groups:
            new_generators = [tuple(g) for g in generators]
            K = create_subgroup_from_permutations(S, new_generators)
            unitaries = derive_unitaries_angle_embedding_analytic(K)
            subgroup_unitaries[k].append({"unitaries": unitaries, "subgroup": K})

    print("Depth:", depth)
    final_results[n_qubits][depth] = {}
    print("  Ansatz ID:", ansatz_id)
    final_results[n_qubits][depth][ansatz_id] = {"original": None, "twirled": {}}
    super_ansatz = Ansatz(ansatz_id, n_qubits, depth)
    ansatz = super_ansatz.get_circuit()
    param_shape = super_ansatz.get_params_shape()

    if param_shape[1] is not None:
        params1 = np.random.uniform(0, 2 * np.pi, param_shape[0])
        params2 = np.random.uniform(0, 2 * np.pi, param_shape[1])
        params = [params1, params2]
        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def circuit(params1, params2):
            ansatz(params1, params2)
            return qml.expval(qml.PauliZ(0))
    else:
        params1 = np.random.uniform(0, 2 * np.pi, param_shape[0])
        params = [params1]
        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def circuit(params1):
            ansatz(params1)
            return qml.expval(qml.PauliZ(0))

    qiskit_circuit = pennylane_to_qiskit(circuit, n_qubits, params=params)
    qiskit_circuit = qiskit_circuit.remove_final_measurements(inplace=False)
    params = qiskit_circuit.parameters
    circuit_descriptor = CircuitDescriptor(qiskit_circuit, params)
    exp = EntanglementCapability(circuit_descriptor, samples=5000)
    original_entanglement = exp.entanglement_capability()
    final_results[n_qubits][depth][ansatz_id]["original"] = original_entanglement

    ansatz_generators = get_ansatz_generators(super_ansatz.get_ansatz())
    for k in tqdm(subgroup_unitaries, desc=f"n_qubits={n_qubits}, depth={depth}, ansatz={ansatz_id}"):
        final_results[n_qubits][depth][ansatz_id]["twirled"][k] = []
        for elem in subgroup_unitaries[k]:
            unitaries = elem["unitaries"]
            twirled_generators = apply_twirling_to_generators(unitaries, ansatz_generators, n_qubits)
            twirled_circuit = QuantumCircuit(n_qubits)
            for i, (gen_matrix, op_wires, op_name, theta, parametrized) in enumerate(ansatz_generators):
                twirled_elem = twirled_generators[i]
                if twirled_elem["gate_name"] == op_name and twirled_elem["wires"] == op_wires:
                    H = twirled_elem['averaged']
                    original_param = None
                    for instr, qargs, cargs in qiskit_circuit.data:
                        occurrence_idx = sum(
                            1
                            for j in range(i + 1)
                            if ansatz_generators[j][2] == op_name and ansatz_generators[j][1] == op_wires
                        )
                        seen = 0
                        for _instr, _qargs, _cargs in qiskit_circuit.data:
                            if _instr.name.lower() == op_name.lower() and [q._index for q in _qargs] == list(op_wires) and len(_instr.params) > 0:
                                seen += 1
                                if seen == occurrence_idx:
                                    original_param = _instr.params[0]
                                    break
                        if original_param is not None:
                            break

                    if original_param is None:
                        param = theta
                    else:
                        param = original_param

                    pauli_op = SparsePauliOp.from_operator(H)
                    evo_gate = PauliEvolutionGate(pauli_op, time=param)
                    twirled_circuit.append(evo_gate, range(n_qubits))
                else:
                    raise ValueError(f"Twirled generator for {op_name} on wires {op_wires} not found when {twirled_elem['gate_name']} and {twirled_elem['wires']}")

            twirled_circuit.remove_final_measurements(inplace=True)
            params = twirled_circuit.parameters
            circuit_descriptor = CircuitDescriptor(twirled_circuit, params)
            exp = EntanglementCapability(circuit_descriptor, samples=5000)
            twirled_entanglement = exp.entanglement_capability()
            print(f"    Twirled entanglement (k={k}): {twirled_entanglement:.4f}")
            final_results[n_qubits][depth][ansatz_id]["twirled"][k].append(twirled_entanglement)

    with open(f"entanglement_results_{n_qubits}_d{depth}_a{ansatz_id}.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entanglement capability experiment for given depth and ansatz_id.")
    parser.add_argument("--depth", type=int, required=True, help="Circuit depth")
    parser.add_argument("--ansatz_id", type=int, required=True, help="Ansatz ID")
    args = parser.parse_args()
    main(args.depth, args.ansatz_id)