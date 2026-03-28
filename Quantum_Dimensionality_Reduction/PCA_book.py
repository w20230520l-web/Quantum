from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import math

circuit = QuantumCircuit(5,5)

# 第一步：初始化两个完全相同的矩阵
circuit.ry(2.148,1)
circuit.ry(2.148,3)
circuit.x(1)
circuit.x(3)
circuit.cry(1.26,1,2)
circuit.cry(1.26,3,4)
circuit.x(1)
circuit.x(3)
circuit.cry(2.49,1,2)
circuit.cry(2.49,3,4)
circuit.barrier()

# d第二步：交换测试
circuit.h(0)
circuit.cswap(0,1,3)
circuit.h(0)

# 测量
circuit.measure(0,0)

# 绘制电路图
fig_circuit = circuit.draw(output='mpl', plot_barriers=False, fold = -1)
plt.show()
backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(circuit,backend)
job = backend.run(transpiled_circuit, shots=20000)
result = job.result()

# 绘制结果图
measurement_result = result.get_counts()
shots = sum(measurement_result.values())
probs = {k: v / shots for k, v in measurement_result.items()}
fig_hist=plot_histogram(probs)
plt.show()
print("Counts:", measurement_result)

