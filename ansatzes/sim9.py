import pennylane as qml

class Sim9:

    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth

    def get_circuit(self):

        def circuit(single_qubit_params):
            for d in range(self.depth):
                for i in range(self.num_qubits):
                    qml.Hadamard(wires=i)
                
                for i in reversed(range(self.num_qubits - 1)):
                    qml.CZ(wires=[i + 1, i])

                for i in range(self.num_qubits):
                    qml.RX(single_qubit_params[d][i], wires=i)

        return circuit

    def get_params_shape(self):
        return (self.depth, self.num_qubits), None