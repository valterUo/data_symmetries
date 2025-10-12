import json
import numpy as np
import pennylane as qml

from ansatz import Ansatz
from twirler.generators import get_ansatz_generators
from twirler.symmetry_groups import create_symmetric_group
from twirler.twirling import apply_twirling_to_generators
from utils import get_subgroup_unitaries

results = {}
n_qubits = 4

S = create_symmetric_group(n_qubits)
with open(f"groups/subgroups_{n_qubits}.json", "r") as f:
    subgroups = json.load(f)

subgroup_unitaries = get_subgroup_unitaries(subgroups, S)

results[n_qubits] = {}

for depth in range(1, 6):
    print(" Depth:", depth)
    results[n_qubits][depth] = {}
    for ansatz_id in range(1, 20):
        print("  Ansatz ID:", ansatz_id)
        super_ansatz = Ansatz(ansatz_id, n_qubits, depth)
        ansatz = super_ansatz.get_ansatz()
        ansatz_generators = get_ansatz_generators(ansatz)

        ansatz_results = {}

        for k in subgroup_unitaries:
            # Precompute full generator matrices once per k
            G_full_list = []
            for gen_matrix, op_wires, op_name, theta, parametrized in ansatz_generators:
                op = qml.Hermitian(gen_matrix, wires=op_wires)
                G_full_list.append(qml.matrix(op, wire_order=range(n_qubits)))

            total = 0.0
            count = 0

            for elem in subgroup_unitaries[k]:
                unitaries = elem["unitaries"]
                subgroup = elem["subgroup"]
                twirled_generators = apply_twirling_to_generators(
                    unitaries, ansatz_generators, n_qubits
                )

                for gen_idx in range(len(ansatz_generators)):
                    G_twirled = twirled_generators[gen_idx]["averaged"]
                    diff = G_full_list[gen_idx] - G_twirled
                    total += np.linalg.norm(diff, ord="fro")
                    count += 1

            avg_norm = total / count
            ansatz_results[k] = avg_norm

        results[n_qubits][depth][ansatz_id] = ansatz_results

with open(f"results/results_projection_onto_symmetric_subspace_{n_qubits}.json", "w") as f:
    json.dump(results, f, indent=4)