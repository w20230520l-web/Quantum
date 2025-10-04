
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
opt = Operator(qc)
