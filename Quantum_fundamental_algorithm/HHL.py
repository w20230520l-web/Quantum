from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import pi
import numpy as np

circuit = QuantumCircuit(4,4)

## 第一步：制备|b>
circuit.x(3)
circuit.barrier(0, 1, 2, 3)

## 第二步：相位估计求特征值
circuit.h(1)
circuit.h(2)
circuit.p(pi*3/4,1)
circuit.cu(-pi/2, -pi/2, pi/2, 0,1,3)
circuit.cx(2,3)
circuit.swap(1, 2)
circuit.h(1)
circuit.cp(-pi/2, 1, 2)
circuit.h(2)

## 第三步：特征值取反
circuit.swap(1,2)
# 受控旋转将特征值提取到辅助量子比特中
circuit.cry(pi/16, 2,0)
circuit.cry(pi/32, 1 ,0)
circuit.swap(1, 2)

# 第四步：逆相位估计
circuit.h(2)
circuit.cp(pi/2, 1, 2)
circuit.h(1)
circuit.swap(1,2)
circuit.cx(2,3)
circuit.cu(-pi/2, pi/2, -pi/2, 0, 1, 3)
circuit.p(-pi*3/4, 1)
circuit.h(1)
circuit.h(2)
circuit.barrier(0, 1, 2, 3)

## 测量
circuit.measure(3,3)
circuit.measure(0,0)

## 绘制电路图
fig_circuit = circuit.draw(output='mpl', plot_barriers=False, fold = -1)
plt.show()
backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(circuit,backend)
job = backend.run(transpiled_circuit, shots=200000)
result = job.result()

#绘制结果图
measurement_result = result.get_counts()
shots = sum(measurement_result.values())
probs = {k: v / shots for k, v in measurement_result.items()}
fig_hist=plot_histogram(probs)
plt.show()
print("Counts:", measurement_result)