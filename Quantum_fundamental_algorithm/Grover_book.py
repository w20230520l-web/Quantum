from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

circuit = QuantumCircuit(3,3)

##
# 第一步
circuit.x(2)
circuit.h([0,1,2])

# 第二步
circuit.ccx(0,1,2)

# 第三步
circuit.h([0,1])
circuit.x([0,1])
circuit.cz(0,1)
circuit.x([0,1])
circuit.h([0,1])

# 测量
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

