import numpy as np
import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pennylane_qiskit import AerDevice

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