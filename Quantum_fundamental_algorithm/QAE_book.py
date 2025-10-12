from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit import transpile
from math import pi
import matplotlib.pyplot as plt
##
circuit = QuantumCircuit(5,5)

# 第一步：制备量子态，制备振幅估计的叠加态
def U():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cry(pi/2,0,1)   # Cry是受控Ry门，作用在0和1之间，当0为1时，对1施加Ry(pi/2)旋转
    circuit.x(0)
    circuit.cry(-pi/2,0,1)
    circuit.x(0)
    circuit.h(0)
    circuit = circuit.to_gate()  # 将量子电路转换为量子门，方便后续直接调用
    circuit.name = "U"
    return circuit

circuit.append(U(), [i+3 for i in range(2)])

# 第二步：对受控比特进行Q门操作

circuit.barrier()
for i in range(3):
    circuit.h(i)

def Q():
    circuit = QuantumCircuit(2)
    circuit.z(0)
    circuit.append(U().inverse(), [i for i in range(2)])
    circuit.x(1)
    circuit.x(0)
    circuit.cz(0,1)  #当两个量子比特都是 |1⟩ 时，相位乘以 -1
    circuit.x(0)
    circuit.x(1)
    circuit.append(U(), [i for i in range(2)])
    circuit = circuit.to_gate()
    circuit.name = "Q"
    c_U = circuit.control()  # 受控版本的量子门
    return c_U

for i in range(3):
    for j in range(2**i):
        circuit.append(Q(), [i] + [m+3 for m in range(2)])


# 第三步：量子傅里叶逆变换
def qft_dagger(n):
    qc =QuantumCircuit(n)
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-pi / 2 ** (j - m), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

circuit.append(qft_dagger(3),[i for i in range(3)])

# 测量
for i in range(3):
    circuit.measure(i, i)

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




