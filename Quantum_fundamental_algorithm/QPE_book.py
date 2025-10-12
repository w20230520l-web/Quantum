from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import math

circuit = QuantumCircuit(4,3)
# 第一步
circuit.x(3)

# 第二步
circuit.barrier()
for qubit in range(3):
    circuit.h(qubit)

# 第三步：受控酉操作
repetitions = 1
for counting_qubit in range(3):
    for i in range(repetitions):
        circuit.cp(math.pi/4, counting_qubit, 3)
    repetitions *= 2

# 第四步：量子傅里叶逆变换
def qft_dagger(qc, n):
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/2**(j-m), m, j)
        qc.h(j)
circuit.barrier()
qft_dagger(circuit, 3)

# 测量
circuit.barrier()
circuit.measure([0,1,2],[0,1,2])

# 绘制电路图
fig_circuit = circuit.draw(output='mpl', plot_barriers=False, fold = -1)
plt.show()
backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(circuit,backend)
job = backend.run(transpiled_circuit, shots=8192)
result = job.result()

#绘制结果图
measurement_result = result.get_counts()
shots = sum(measurement_result.values())
probs = {k: v / shots for k, v in measurement_result.items()}
fig_hist=plot_histogram(probs)
plt.show()
print("Counts:", measurement_result)




