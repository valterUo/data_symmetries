import pennylane as qml
from pennylane import numpy as np

class Sim8:

    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth

    def get_circuit(self):

        def circuit(single_qubit_params, two_qubit_params):
            for d in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RX(single_qubit_params[d][i][0], wires=i)
                    qml.RZ(single_qubit_params[d][i][1], wires=i)
                
                for i in range(0, self.num_qubits - 1, 2):
                    qml.CRX(two_qubit_params[d][i], wires=[i + 1, i])
                        
                for i in range(self.num_qubits):
                    qml.RX(single_qubit_params[d][i][2], wires=i)
                    qml.RZ(single_qubit_params[d][i][3], wires=i)
                
                for i in range(1, self.num_qubits - 1, 2):
                    qml.CRX(two_qubit_params[d][i], wires=[i + 1, i])

        return circuit

    def get_params_shape(self):
        return (self.depth, self.num_qubits, 4), (self.depth, self.num_qubits)