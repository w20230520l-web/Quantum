import numpy as np
import matplotlib.pyplot as plt
from math import pi, asin
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator

# --- 0. 准备矩阵数据 ---
# 定义矩阵 A (4x4)
A = 0.25 * np.array([
    [15, 9, 5, -3],
    [9, 15, 3, -5],
    [5, 3, 15, -9],
    [-3, -5, -9, 15]
])
# 定义向量 b (归一化)
b_vec = np.array([1, -1, -1, -1])
b_norm = np.linalg.norm(b_vec)
b_state = b_vec / b_norm

# 计算演化算子 U = exp(i*A*t)
t = 2 * pi / 16

# --- 构建电路 ---
circuit = QuantumCircuit(7, 3)
anc = 0
clock = [1, 2, 3, 4]
target = [5, 6]

## 第一步：制备|b>
circuit.initialize(b_state, target)
circuit.barrier()

## 第二步：相位估计 (QPE)
for q in clock:
    circuit.h(q)

for i in range(4):
    power = 2 ** i
    U_matrix = expm(1j * A * t * power)
    cu_gate = Operator(U_matrix).to_instruction()
    cu_gate.label = f"CU^{power}"
    cu_gate = cu_gate.control(1)
    circuit.append(cu_gate, [clock[i]] + target)

# 逆 QFT
n_clock = 4
for i in range(n_clock // 2):
    circuit.swap(clock[i], clock[n_clock - 1 - i])
for i in range(n_clock):
    for j in range(i):
        circuit.cp(-pi / (2 ** (i - j)), clock[j], clock[i])
    circuit.h(clock[i])

circuit.barrier()

## 第三步：特征值取反 (受控旋转)
circuit.cry(2 * asin(1 / 1), clock[0], anc)
circuit.cry(2 * asin(1 / 2), clock[1], anc)
circuit.cry(2 * asin(1 / 4), clock[2], anc)
circuit.cry(2 * asin(1 / 8), clock[3], anc)

circuit.barrier()

## 第四步：逆相位估计 (Uncompute)
for i in range(n_clock - 1, -1, -1):
    circuit.h(clock[i])
    for j in range(i - 1, -1, -1):
        circuit.cp(pi / (2 ** (i - j)), clock[j], clock[i])
for i in range(n_clock // 2):
    circuit.swap(clock[i], clock[n_clock - 1 - i])

for i in range(4 - 1, -1, -1):
    power = 2 ** i
    U_matrix = expm(1j * A * t * power)
    cu_gate_inv = Operator(U_matrix).to_instruction().inverse()
    cu_gate_inv.label = f"CU^{power}_dag"
    cu_gate_inv = cu_gate_inv.control(1)
    circuit.append(cu_gate_inv, [clock[i]] + target)

for q in clock:
    circuit.h(q)

circuit.barrier()

## 测量
circuit.measure(anc, 0)
circuit.measure(target[0], 1)
circuit.measure(target[1], 2)

## 绘制电路图
print("正在生成量子线路图...")
fig_circuit = circuit.draw(output='mpl',
                           fold=-1,
                           scale=0.7,
                           style={
                               'compress': True,
                               'fontsize': 10,
                               'subfontsize': 8,
                               'figwidth': 15
                           },
                           filename="hhl_compact.png")
plt.show()

## 运行模拟
print("正在运行模拟...")
backend = Aer.get_backend('aer_simulator')
transpiled_circuit = transpile(circuit, backend)
job = backend.run(transpiled_circuit, shots=200000)
result = job.result()
counts = result.get_counts()
measurement_result = counts


# --- 修改重点开始：从计数改为概率 ---

# 1. 筛选辅助比特为 1 的结果 (还是统计 Counts)
valid_counts = {}
total_success = 0
for k, v in counts.items():
    k = k.replace(" ", "")
    meas_anc = k[-1]
    meas_target = k[:-1]

    if meas_anc == '1':
        valid_counts[meas_target] = valid_counts.get(meas_target, 0) + v
        total_success += v

if total_success > 0:
    # 2. 将 Counts 转换为 Probabilities
    valid_probs = {}
    all_states = ['00', '01', '10', '11']

    for state in all_states:
        # 如果某个态测到了，计算 count/total；没测到则为 0.0
        if state in valid_counts:
            valid_probs[state] = valid_counts[state] / total_success
        else:
            valid_probs[state] = 0.0

    # 3. 绘制概率直方图
    # plot_histogram 也可以接受字典 {Label: Probability}
    print("\n正在生成概率分布图...")
    fig_hist = plot_histogram(valid_probs)
    plt.ylabel("Probability ")
    plt.show()
    print("Counts:", measurement_result)

    print("\n=== 求解结果 (概率形式) ===")
    print(f"成功样本数: {total_success}")
    print("解向量概率分布:")
    for state in sorted(valid_probs.keys()):
        print(f"|{state}>: {valid_probs[state]:.4f}")
else:
    print("未能获得有效结果，请增加 shots 或检查旋转角度。")