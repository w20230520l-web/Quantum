from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import math
##
# 计算arccos
result1 = math.acos(0.999)
result2 = math.acos(0.339)

circuit = QuantumCircuit(2,2)
circuit.h(0)
circuit.cry(2*result1, 0,1)
circuit.x(0)
circuit.cry(2*result2,0,1)
circuit.x(0)
circuit.h(0)

circuit.measure(0,0)
fig_circuit = circuit.draw(output='mpl')
plt.show()
backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(circuit,backend)
job = backend.run(transpiled_circuit, shots=8192)
result = job.result()
#绘制结果图z
measurement_result = result.get_counts()
shots = sum(measurement_result.values())
probs = {k: v / shots for k, v in measurement_result.items()}
fig_hist=plot_histogram(probs)
plt.show()
print("Counts:", measurement_result)
