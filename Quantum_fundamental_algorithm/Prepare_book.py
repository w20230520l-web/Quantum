from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
##
circuit = QuantumCircuit(2,2)

#第一步
circuit.ry(1.939,1)

#第二步
circuit.x(1)
circuit.cry(1.571,1,0)
circuit.x(1)

#第三步
circuit.cry(0.490,1,0)

#测量
circuit.measure(0,0)
circuit.measure(1,1)

#绘制电路图
fig_circuit = circuit.draw(output='mpl')
plt.show()
backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(circuit,backend)
job = backend.run(transpiled_circuit, shots=20000)
result = job.result()

#绘制结果图
measurement_result = result.get_counts()
shots = sum(measurement_result.values())
probs = {k: v / shots for k, v in measurement_result.items()}

fig_hist=plot_histogram(probs)
plt.show()
print("Counts:", measurement_result)