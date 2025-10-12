from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix, partial_trace
import numpy as np
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


def show_single_qubit_phases(qc):
    """
    打印电路输出态中每个量子位 (|0> + e^{iφ}|1>)/√2 的相对相位 φ。
    若该位不是上述纯态（例如被纠缠或完全对角），则 off-diagonal 会很小/为 0。
    """
    qc_nom = qc.remove_final_measurements(inplace=False)  # 去掉末尾测量
    rho_full = DensityMatrix.from_instruction(qc_nom)
    n = qc.num_qubits
    phases = []
    for q in range(n):
        # 对除了 q 以外的比特做偏迹，得到该比特的 2x2 密度矩阵
        rest = [i for i in range(n) if i != q]
        rho_q = partial_trace(rho_full, rest).data  # 2x2 numpy 数组
        off = rho_q[0, 1]
        if np.isclose(abs(off), 0.0, atol=1e-12):
            phases.append(None)  # 该位没有明确定义的相位（完全对角/高度混合）
        else:
            phi = (-np.angle(off)) % (2*np.pi)
            phases.append(phi)
            print(f"q{q}: φ ≈ {phi:.6f} rad  ({phi/np.pi:.4f} π)")
    return phases

phases = show_single_qubit_phases(circuit)


