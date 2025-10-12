import numpy as np
import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pennylane_qiskit import AerDevice

from twirler.induced_representation import derive_unitaries_angle_embedding_analytic
from twirler.symmetry_groups import create_subgroup_from_permutations

def get_angle_encoding_unitary_Z(data):

    U = np.array([[1.0]], dtype=complex)
    
    for i in range(len(data)):

        RZ = np.array([
            [np.exp(-1j * data[i] / 2), 0],
            [0, np.exp(1j * data[i] / 2)]
        ], dtype=complex)
        
        U = np.kron(U, RZ)
    
    return U

def angle_encoding_unitary_mixed(x):

    U = np.array([[1.+0j]])

    for theta in x:
        RZ = np.array([[np.exp(-1j*theta[0]/2), 0],
                       [0, np.exp(1j*theta[0]/2)]], dtype=complex)
        
        RX = np.array([[np.cos(theta[1]/2), -1j*np.sin(theta[1]/2)],
                       [-1j*np.sin(theta[1]/2), np.cos(theta[1]/2)]], dtype=complex)
        
        RY = np.array([[np.cos(theta[2]/2), -np.sin(theta[2]/2)],
                       [np.sin(theta[2]/2),  np.cos(theta[2]/2)]], dtype=complex)
        
        U1 = RZ @ RX @ RY
        U = np.kron(U, U1)
    return U


def angle_encoding_unitary_Z_X(x):

    U = np.array([[1.+0j]])

    for theta in x:
        RZ = np.array([[np.exp(-1j*theta[0]/2), 0],
                       [0, np.exp(1j*theta[0]/2)]], dtype=complex)
        
        RX = np.array([[np.cos(theta[1]/2), -1j*np.sin(theta[1]/2)],
                       [-1j*np.sin(theta[1]/2), np.cos(theta[1]/2)]], dtype=complex)
        
        U1 = RZ @ RX
        U = np.kron(U, U1)
    return U


def pennylane_to_qiskit(circuit, n_qubits, params = None, symbolic_params = True):
    qiskit_device = AerDevice(wires=n_qubits)
    qnode = qml.QNode(circuit.func, qiskit_device)
    if params is not None:
        qnode(*params)
    else:
        qnode()
    qiskit_circuit = qiskit_device._circuit

    if symbolic_params:
        # Create a new circuit with symbolic parameters
        new_qc = QuantumCircuit(n_qubits, n_qubits)
        param_mapping = {}  # Store parameters for reuse

        for instr, qubits, clbits in qiskit_circuit.data:
            new_params = []
            
            # Replace constant parameters with symbolic ones
            for param in instr.params:
                if isinstance(param, (float, int)):  # Detect constants
                    if param not in param_mapping:
                        param_mapping[param] = Parameter(f"x{len(param_mapping)}")
                    new_params.append(param_mapping[param])
                else:
                    new_params.append(param)

            new_qc.append(instr.__class__(*new_params), qubits, clbits)
    else:
        new_qc = qiskit_circuit

    return new_qc

from qiskit import QuantumCircuit

def circuit_stats(circuit: QuantumCircuit, verbose: bool = False):
    """Return basic statistics about a Qiskit QuantumCircuit."""
    stats = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "num_ops": len(circuit.data),
        "gate_counts": circuit.count_ops(),
        "number_of_parameters": len(circuit.parameters)
    }
    
    if verbose:
        print("Quantum Circuit Statistics:")
        print(f"  Qubits:       {stats['num_qubits']}")
        print(f"  Clbits:       {stats['num_clbits']}")
        print(f"  Depth:        {stats['depth']}")
        print(f"  Size:         {stats['size']}")
        print(f"  Operations:   {stats['num_ops']}")
        print(f"  Gate counts:  {dict(stats['gate_counts'])}")
        print(f"  Parameters:   {stats['number_of_parameters']}")
    
    return stats

def find_original_param(qiskit_circuit, ansatz_generators, i, op_name, op_wires, theta):
    """
    Find the original parameter for the ith generator with given op_name and op_wires in qiskit_circuit.
    If not found, return theta.
    """
    original_param = None
    # Find the nth occurrence (1-based) of this gate on these wires
    occurrence_idx = sum(
        1
        for j in range(i + 1)
        if ansatz_generators[j][2] == op_name and ansatz_generators[j][1] == op_wires
    )
    seen = 0
    for instr, qargs, cargs in qiskit_circuit.data:
        if instr.name.lower() == op_name.lower() and [q._index for q in qargs] == list(op_wires) and len(instr.params) > 0:
            seen += 1
            if seen == occurrence_idx:
                original_param = instr.params[0]
                break
    if original_param is None:
        return theta
    else:
        return original_param
    

def get_subgroup_unitaries(subgroups, S):
    subgroup_unitaries = {}
    for k, groups in subgroups.items():
        subgroup_unitaries[k] = []
        for generators in groups:
            new_generators = [tuple(g) for g in generators]
            K = create_subgroup_from_permutations(S, new_generators)
            unitaries = derive_unitaries_angle_embedding_analytic(K)
            subgroup_unitaries[k].append({"unitaries": unitaries, "subgroup": K})
    return subgroup_unitaries
