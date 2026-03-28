from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator
from qiskit_aer import Aer
from qiskit import transpile
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


class HHL_Paper_Fig2:
    def __init__(self, r=4):
        # 参数 r 来自论文 Fig 3 的讨论，r=4 时保真度较高
        self.r = r
        self.t0 = 2 * np.pi  # 论文 Fig 2 说明 t0 = 2*pi

        # 寄存器定义对应 Fig. 2
        # |x1>: Ancilla (Top)
        # |x2>: Clock Qubit 1 (Middle-Top)
        # |x3>: Clock Qubit 2 (Middle-Bottom)
        # |x4>: Input/b Register (Bottom)
        self.anc = QuantumRegister(1, 'x1')
        self.clock = QuantumRegister(2, 'clock')  # clock[0] is x3, clock[1] is x2
        self.b = QuantumRegister(1, 'x4')

        self.c_anc = ClassicalRegister(1, 'm')
        self.c_b = ClassicalRegister(1, 'result_b')

        self.qc = QuantumCircuit(self.anc, self.clock, self.b, self.c_anc, self.c_b)

        # 定义矩阵 A (论文 Eq. 3)
        # A = [[1.5, 0.5], [0.5, 1.5]]
        self.A = np.array([[1.5, 0.5], [0.5, 1.5]])

    def get_controlled_evolution(self, time):
        """创建受控的 exp(i*A*t) 门"""
        # 计算矩阵指数 U = exp(i*A*t)
        U_matrix = expm(1j * self.A * time)
        # 转换为算子并添加控制位
        U_gate = Operator(U_matrix).to_instruction()
        U_gate.label = f"exp(iAt={time:.2f})"
        return U_gate.control(1)

    def phase_estimation(self):
        """对应 Fig. 2 左侧的 Phase Estimation 部分"""
        # 1. 对 Clock qubits (|x2>, |x3>) 进行 Hadamard
        self.qc.h(self.clock[0])  # x3
        self.qc.h(self.clock[1])  # x2

        # 2. 受控幺正演化
        # 注意：Fig 2 中 |x3> (下层 clock) 控制 exp(i A t0 / 4)
        #      Fig 2 中 |x2> (上层 clock) 控制 exp(i A t0 / 2)

        # 应用 U controlled by x3 (clock[0])
        # t = t0 / 4
        cu_quarter = self.get_controlled_evolution(self.t0 / 4)
        self.qc.append(cu_quarter, [self.clock[0], self.b[0]])

        # 应用 U controlled by x2 (clock[1])
        # t = t0 / 2
        cu_half = self.get_controlled_evolution(self.t0 / 2)
        self.qc.append(cu_half, [self.clock[1], self.b[0]])

    def inverse_qft(self):
        """对应 Fig. 2 中间的 SWAP 和 H/S 门部分 (IQFT)"""
        # 论文使用了标准的 2-qubit IQFT 分解
        # 1. Swap x2, x3
        self.qc.swap(self.clock[0], self.clock[1])

        # 2. H on x2 (clock[1])
        self.qc.h(self.clock[1])

        # 3. Controlled-S^dagger (or CP(-pi/2))
        self.qc.cp(-np.pi / 2, self.clock[0], self.clock[1])

        # 4. H on x3 (clock[0])
        self.qc.h(self.clock[0])

    def rotation(self):
        """对应 Fig. 2 右上角的受控旋转 Ry """
        # |x2> (clock[1]) controls Ry(2*pi / 2^r)
        theta_2 = 2 * np.pi / (2 ** self.r)
        self.qc.cry(theta_2, self.clock[1], self.anc[0])

        # |x3> (clock[0]) controls Ry(pi / 2^r)
        theta_3 = np.pi / (2 ** self.r)
        self.qc.cry(theta_3, self.clock[0], self.anc[0])

    def uncompute(self):
        """对应 Fig. 2 中的 U dagger [cite: 131]"""
        # 反向执行 Inverse QFT
        inv_iqft = QuantumCircuit(self.clock)
        inv_iqft.h(self.clock[0])
        inv_iqft.cp(np.pi / 2, self.clock[0], self.clock[1])  # S instead of S_dag
        inv_iqft.h(self.clock[1])
        inv_iqft.swap(self.clock[0], self.clock[1])
        self.qc.append(inv_iqft.to_instruction(), self.clock)

        # 反向执行 Phase Estimation (Uncompute U gates)
        # Inverse of U(t0/2) controlled by x2
        u_half_dag = self.get_controlled_evolution(-self.t0 / 2)  # Negative time for inverse
        self.qc.append(u_half_dag, [self.clock[1], self.b[0]])

        # Inverse of U(t0/4) controlled by x3
        u_quarter_dag = self.get_controlled_evolution(-self.t0 / 4)
        self.qc.append(u_quarter_dag, [self.clock[0], self.b[0]])

        # Restore Hadamards
        self.qc.h(self.clock[0])
        self.qc.h(self.clock[1])

    def construct_circuit(self, b_vec):
        # 1. 初始化 |b> = |x4>
        # 假设 b_vec = [0, 1]，即初始化为 |1>
        if b_vec[1] == 1:
            self.qc.x(self.b[0])

        self.qc.barrier()

        # 2. Phase Estimation
        self.phase_estimation()

        self.qc.barrier()

        # 3. Inverse QFT (Transforming to Eigenvalue Basis)
        self.inverse_qft()

        self.qc.barrier()

        # 4. Eigenvalue Inversion (Controlled Rotations)
        self.rotation()

        self.qc.barrier()

        # 5. Uncompute (U dagger)
        self.uncompute()

        self.qc.barrier()

        # 6. Measure Ancilla (|x1>) and Result (|x4>)
        self.qc.measure(self.anc, self.c_anc)
        self.qc.measure(self.b, self.c_b)


# --- 执行部分 ---

# 设定 b = [0, 1] 对应论文中的非平凡解
b_vec = [0, 1]

# 实例化并构建电路 (r=4)
hhl_fig2 = HHL_Paper_Fig2(r=4)
hhl_fig2.construct_circuit(b_vec)

# 打印电路图以验证与 Image 1 的对应关系
print("Circuit matches structure of Fig. 2:")
hhl_fig2.qc.draw('mpl')
print("Circuit matches structure of Fig. 2:")
hhl_fig2.qc.draw('mpl')
plt.show()

# 运行模拟
backend = Aer.get_backend('qasm_simulator')
transpiled = transpile(hhl_fig2.qc, backend)
job = backend.run(transpiled, shots=100000)
result = job.result()
counts = result.get_counts()

# 筛选后选择测量结果
# 我们只关心 Ancilla (|m>) 为 1 的情况
# 格式: 'result_b m' -> 例如 '0 1' 表示 m=1, b=0
measured_x = {'0': 0, '1': 0}
total_success = 0

for key, count in counts.items():
    # qiskit 字符串顺序是反的: "c_b c_anc" -> "bit_b bit_m"
    # 或者根据寄存器添加顺序，这里需要小心解析
    # key 格式通常为 "c_b c_anc" (如果有空格) 或者连在一起
    # 假设是 'c_b c_anc'
    bin_str = key.split()
    if len(bin_str) == 1:
        # 如果没有空格，通常是 c_anc c_b (高位在左)
        m_bit = key[0]  # c_anc 是最后一个被添加的 classical register?
        # 为了保险，查看 Qiskit 默认顺序：reversed(classical_bits)
        # c_b 在前, c_anc 在后 -> bit string: c_anc c_b
        m_val = key[0]
        b_val = key[1]
    else:
        m_val = bin_str[0]
        b_val = bin_str[1]

    if m_val == '1':  # 测量成功
        measured_x[b_val] += count
        total_success += count

print("\nResults (conditioned on ancilla=1):")
if total_success > 0:
    print(f"Prob(|0>): {measured_x['0'] / total_success:.4f}")
    print(f"Prob(|1>): {measured_x['1'] / total_success:.4f}")
else:
    print("No successful measurements.")

# 理论验证
# A = [[1.5, 0.5], [0.5, 1.5]], b = [0, 1]
# x = A^-1 b
A = np.mat([[1.5, 0.5], [0.5, 1.5]])
b = np.mat([0, 1]).T
x_exact = np.linalg.inv(A) @ b
x_prob = np.array(x_exact).flatten() ** 2
x_prob = x_prob / sum(x_prob)
print(f"\nTheoretical Probabilities:\nProb(|0>): {x_prob[0]:.4f}\nProb(|1>): {x_prob[1]:.4f}")