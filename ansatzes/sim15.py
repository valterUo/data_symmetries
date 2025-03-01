import pennylane as qml

class Sim15:

    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth

    def get_circuit(self):

        def circuit(single_qubit_params):
            for d in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(single_qubit_params[d][i][0], wires=i)

                qml.CNOT(wires=[self.num_qubits - 1, 0])
                
                for i in reversed(range(self.num_qubits - 1)):
                    qml.CNOT(wires=[i, i + 1])
                
                for i in range(self.num_qubits):
                    qml.RY(single_qubit_params[d][i][1], wires=i)
                
                qml.CNOT(wires=[self.num_qubits - 1, self.num_qubits - 2])
                qml.CNOT(wires=[0, self.num_qubits - 1])
                
                for i in range(self.num_qubits - 2):
                    qml.CNOT(wires=[i, i + 1])

        return circuit

    def get_params_shape(self):
        return (self.depth, self.num_qubits, 2), None