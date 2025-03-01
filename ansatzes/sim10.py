import pennylane as qml

class Sim10:

    def __init__(self, num_qubits, depth):
        self.num_qubits = num_qubits
        self.depth = depth

    def get_circuit(self):

        def circuit(single_qubit_params):
            for i in range(self.num_qubits):
                qml.RY(single_qubit_params[0][i], wires=i)

            for d in range(1, self.depth + 1):
                for i in reversed(range(self.num_qubits - 1)):
                    qml.CZ(wires=[i + 1, i])
                
                qml.CZ(wires=[0, self.num_qubits - 1])
                
                for i in range(self.num_qubits):
                    qml.RY(single_qubit_params[d][i], wires=i)

        return circuit

    def get_params_shape(self):
        return (self.depth + 1, self.num_qubits), None