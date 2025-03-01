from ansatzes.sim1 import Sim1
from ansatzes.sim2 import Sim2
from ansatzes.sim3 import Sim3
from ansatzes.sim4 import Sim4
from ansatzes.sim5 import Sim5
from ansatzes.sim6 import Sim6
from ansatzes.sim7 import Sim7
from ansatzes.sim8 import Sim8
from ansatzes.sim9 import Sim9
from ansatzes.sim10 import Sim10
from ansatzes.sim11 import Sim11
from ansatzes.sim12 import Sim12
from ansatzes.sim13 import Sim13
from ansatzes.sim14 import Sim14
from ansatzes.sim15 import Sim15
from ansatzes.sim16 import Sim16
from ansatzes.sim17 import Sim17
from ansatzes.sim18 import Sim18
from ansatzes.sim19 import Sim19

class Ansatz:

    def __init__(self, id, num_qubits, depth):
        if id == 1:
            self.ansatz = Sim1(num_qubits, depth)
        elif id == 2:
            self.ansatz = Sim2(num_qubits, depth)
        elif id == 3:
            self.ansatz = Sim3(num_qubits, depth)
        elif id == 4:
            self.ansatz = Sim4(num_qubits, depth)
        elif id == 5:
            self.ansatz = Sim5(num_qubits, depth)
        elif id == 6:
            self.ansatz = Sim6(num_qubits, depth)
        elif id == 7:
            self.ansatz = Sim7(num_qubits, depth)
        elif id == 8:
            self.ansatz = Sim8(num_qubits, depth)
        elif id == 9:
            self.ansatz = Sim9(num_qubits, depth)
        elif id == 10:
            self.ansatz = Sim10(num_qubits, depth)
        elif id == 11:
            self.ansatz = Sim11(num_qubits, depth)
        elif id == 12:
            self.ansatz = Sim12(num_qubits, depth)
        elif id == 13:
            self.ansatz = Sim13(num_qubits, depth)
        elif id == 14:
            self.ansatz = Sim14(num_qubits, depth)
        elif id == 15:
            self.ansatz = Sim15(num_qubits, depth)
        elif id == 16:
            self.ansatz = Sim16(num_qubits, depth)
        elif id == 17:
            self.ansatz = Sim17(num_qubits, depth)
        elif id == 18:
            self.ansatz = Sim18(num_qubits, depth)
        elif id == 19:
            self.ansatz = Sim19(num_qubits, depth)
        
    def get_circuit(self):
        return self.ansatz.get_circuit()
    
    def get_params_shape(self):
        return self.ansatz.get_params_shape()

        