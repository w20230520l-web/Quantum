import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 经典有限元建模与精确求解 (参考你的截图逻辑)
# ==========================================
def build_three_node_bar_quantum_ready():
    """
    构建 3杆4节点 线性弹簧模型。
    节点0固定，节点3受集中力。消去节点0后，补充 dummy 节点使其成为 4x4 矩阵，以适应2个量子比特。
    """
    K = np.array((
        ( 2.0, -1.0,  0.0,  0.0),
        (-1.0,  2.0, -1.0,  0.0),
        ( 0.0, -1.0,  1.0,  0.0),
        ( 0.0,  0.0,  0.0,  1.0)
    ))
    f = np.array((0.0, 0.0, 1.0, 0.0))
    return K, f

def classical_solve_and_energy(K, f):
    """经典求解器并计算应变能"""
    u = np.linalg.solve(K, f)
    # 计算应变能: U = 0.5 * u^T * K * u
    strain_energy = 0.5 * np.dot(u, np.dot(K, u))
    return u, strain_energy

# 获取经典解
K_matrix, f_vector = build_three_node_bar_quantum_ready()
classical_u, classical_energy = classical_solve_and_energy(K_matrix, f_vector)
classical_normalized = classical_u / np.linalg.norm(classical_u)

print("--- 经典有限元求解结果 ---")
print(f"经典位移解: {classical_u}")
print(f"系统应变能: {classical_energy:.4f}")
print(f"归一化位移解 (用于与量子态对比): {classical_normalized}\n")


# ==========================================
# 2. 自动化泡利分解 (核心升级部分)
# ==========================================
print("正在自动分解刚度矩阵...")
# 使用 PennyLane 内置的分解器将矩阵转化为 Pauli 算符的组合
H = qml.pauli_decompose(K_matrix)

# 提取系数和观测算符
coeffs, obs = H.terms()
c = np.array(coeffs)

# 提取泡利字符串 (例如将算符转化为 'IX', 'ZI' 等字符串)，方便后续动态构建电路
wire_map = dict()
wire_map[0] = 0
wire_map[1] = 1
pauli_strings = list()
for op in obs:
    pauli_strings.append(qml.pauli.pauli_word_to_string(op, wire_map=wire_map))

print("--- 刚度矩阵的泡利分解结果 ---")
for i in range(len(c)):
    print(f"项 {i}: 系数 = {c[i]:.2f}, 泡利操作 = {pauli_strings[i]}")

# ==========================================
# 3. 动态构建受控量子门操作
# ==========================================
n_qubits = 2
ancilla_idx = 2
dev = qml.device("default.qubit", wires=n_qubits + 1)


def apply_pauli_string(p_str, control_wire=None):
    """
    通用解析器：读取 'IX', 'ZZ' 等字符串，自动在电路上施加对应的受控/非受控量子门。
    这彻底取代了之前繁琐的 if/elif 穷举。
    """
    for wire, char in enumerate(p_str):
        if char == 'I':
            continue
        elif char == 'X':
            if control_wire is not None:
                qml.CNOT(wires=(control_wire, wire))
            else:
                qml.PauliX(wires=wire)
        elif char == 'Y':
            if control_wire is not None:
                qml.CY(wires=(control_wire, wire))
            else:
                qml.PauliY(wires=wire)
        elif char == 'Z':
            if control_wire is not None:
                qml.CZ(wires=(control_wire, wire))
            else:
                qml.PauliZ(wires=wire)

def CA(idx):
    """施加受控的 A_l 操作"""
    apply_pauli_string(pauli_strings[idx], control_wire=ancilla_idx)

def apply_A(idx):
    """施加无控制的 A_l 操作"""
    apply_pauli_string(pauli_strings[idx], control_wire=None)

# ==========================================
# 4. 量子态制备与拟设 (与之前相同)
# ==========================================
def U_b():
    """制备载荷向量 |b> = (0, 0, 1, 0)^T，对应量子态 |10>"""
    qml.PauliX(wires=0)

def variational_block(weights):
    """硬件高效拟设"""
    n_layers = len(weights) - 1
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)
        qml.CNOT(wires=(0, 1))
    for i in range(n_qubits):
        qml.RY(weights[-1, i], wires=i)


# ==========================================
# 5. 哈达玛测试与代价函数
# ==========================================
@qml.qnode(dev, diff_method="parameter-shift")
def hadamard_test_beta(weights, l, lp):
    qml.Hadamard(wires=ancilla_idx)
    variational_block(weights)
    CA(l)
    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(ancilla_idx))


@qml.qnode(dev, diff_method="parameter-shift")
def hadamard_test_mu(weights, l):
    qml.Hadamard(wires=ancilla_idx)

    def W():
        variational_block(weights)
        apply_A(l)
        qml.adjoint(U_b)()

    qml.ctrl(W, control=ancilla_idx)()
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(ancilla_idx))


def compute_cost(weights):
    global_norm = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            global_norm += c[l] * c[lp] * hadamard_test_beta(weights, l, lp)

    mu_sum = 0.0
    for l in range(len(c)):
        mu_sum += c[l] * hadamard_test_mu(weights, l)

    global_overlap = mu_sum ** 2
    return 1.0 - (global_overlap / global_norm)


# ==========================================
# 6. 经典优化训练循环
# ==========================================
np.random.seed(42)
n_layers = 3
w = 0.01 * np.random.randn(n_layers + 1, n_qubits, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.05)
steps = 200
cost_history = list()

print("\n开始 VQLS 量子有限元训练...")
for it in range(steps):
    w, cost = opt.step_and_cost(compute_cost, w)
    cost_history.append(cost)
    if it % 10 == 0 or it == steps - 1:
        print(f"迭代步数 {it:3d} | 代价函数 (Cost) = {cost:.6f}")


# ==========================================
# 7. 结果对比与提取
# ==========================================
dev_state = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev_state)
def get_quantum_state(weights):
    variational_block(weights)
    return qml.state()

quantum_solution = np.real(get_quantum_state(w))

print("\n--- 最终结果对比 ---")
print(f"经典有限元精确解 (归一化): {classical_normalized}")
print(f"量子变分拟设生成解 (量子态): {quantum_solution}")

plt.plot(cost_history, "b-o")
plt.ylabel("Cost Function")
plt.xlabel("Optimization Steps")
plt.title("Q-FEM vs Classical FEM Convergence")
plt.yscale("log")
plt.show()