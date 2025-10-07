from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import math

circuit = QuantumCircuit(4,4)

# 第一步
circuit.x([1,3])

# 第二步~第五步， 定义函数qft_rotations
circuit.barrier()   # barrier是用来分隔电路的不同部分，方便对于每一部分进行整体编译计算
def qft_rotations(circuit, n, nu):
    if n == 0:
        return circuit
    n -= 1
    nu += 1
    circuit.h(3-n)
    for qubit in range(n):
        circuit.cp(math.pi/2**(qubit+1), qubit+nu, 3-n)    # cp是受控相位门，作用在qubit+nu和3-n之间，作用角度为pi/2^(qubit+1)
    circuit.barrier()
    qft_rotations(circuit,n,nu)

# 调用函数qft_rotations
qft_rotations(circuit,4,0)

# 第六步， 交换量子位
circuit.swap(0,3)
circuit.swap(1,2)

# 测量
circuit.measure([0,1,2,3],[0,1,2,3])

# 绘制电路图
fig_circuit = circuit.draw(output='mpl', plot_barriers=False)
plt.show()


