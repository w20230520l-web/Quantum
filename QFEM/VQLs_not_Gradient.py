import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import LogFormatterExponent, LogLocator

# ==========================================
# 1. 经典有限元建模
# ==========================================
def build_three_node_bar():
    K = np.array((
        (2.0, -1.0, 0.0, 0.0),
        (-1.0, 2.0, -1.0, 0.0),
        (0.0, -1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ))
    f = np.array((0.0, 0.0, 1.0, 0.0))
    return K, f

K_matrix, f_vector = build_three_node_bar()
classical_u = np.linalg.solve(K_matrix, f_vector)
classical_normalized = classical_u / np.linalg.norm(classical_u)

print("--- 经典有限元求解结果 ---")
print(f"经典位移解: {classical_u}")
print(f"归一化位移解 : {classical_normalized}\n")

# ==========================================
# 2. 自动化泡利分解
# ==========================================
print("正在分解刚度矩阵...")
H = qml.pauli_decompose(K_matrix)
coeffs, obs = H.terms()
c = np.array(coeffs)

wire_map = dict()
wire_map = 0
wire_map[1] = 1
pauli_strings = list()
for op in obs:
    pauli_strings.append(qml.pauli.pauli_word_to_string(op, wire_map=wire_map))

# ==========================================
# 3. 动态构建量子门操作
# ==========================================
n_qubits = 2
ancilla_idx = 2
dev = qml.device("default.qubit", wires=n_qubits + 1)

def apply_pauli_string(p_str, control_wire=None):
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
    apply_pauli_string(pauli_strings[idx], control_wire=ancilla_idx)

def apply_A(idx):
    apply_pauli_string(pauli_strings[idx], control_wire=None)

def U_b():
    qml.PauliX(wires=0)

def variational_block(weights):
    n_layers = len(weights) - 1
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)
        qml.CNOT(wires=(0, 1))
    for i in range(n_qubits):
        qml.RY(weights[-1, i], wires=i)

# ==========================================
# 4. 哈达玛测试簇
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

@qml.qnode(dev, diff_method="parameter-shift")
def hadamard_test_local_z(weights, l, lp, j):
    qml.Hadamard(wires=ancilla_idx)
    variational_block(weights)
    CA(l)
    qml.adjoint(U_b)()
    qml.CZ(wires=(ancilla_idx, j))
    U_b()
    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(ancilla_idx))

# ==========================================
# 5. 定义双代价函数
# ==========================================
def cost_local(weights):
    global_norm = 0.0
    beta_vals = dict()
    for l in range(len(c)):
        for lp in range(len(c)):
            val = hadamard_test_beta(weights, l, lp)
            beta_vals[(l, lp)] = val
            global_norm += c[l] * c[lp] * val

    local_overlap = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            z_sum = 0.0
            for j in range(n_qubits):
                z_sum += hadamard_test_local_z(weights, l, lp, j)
            term_overlap = 0.5 * (n_qubits * beta_vals[(l, lp)] + z_sum)
            local_overlap += c[l] * c[lp] * term_overlap

    return 1.0 - (local_overlap / (n_qubits * global_norm))

def cost_global(weights):
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

opt = qml.GradientDescentOptimizer(stepsize=0.05)
steps = 100

history_L = list()
history_G = list()

print("\n开始 VQLS 量子有限元训练...")
for it in range(steps):
    w, c_L = opt.step_and_cost(cost_local, w)
    c_G = cost_global(w)
    history_L.append(c_L)
    history_G.append(c_G)
    if it % 10 == 0 or it == steps - 1:
        print(f"Step {it:3d} | Local Cost = {c_L:.5f} | Global Cost = {c_G:.5f}")

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
print(f"经典有限元精确解: {classical_normalized}")
print(f"量子变分拟设生成解: {quantum_solution}")

# ==========================================
# 8. 双轨对比平滑绘图 (完全套用你提供的高级美化模板)
# ==========================================
x_data = np.arange(len(history_L))
y_data_L = np.array(history_L)
y_data_G = np.array(history_G)

# 确保数据在 log 范围内
y_data_L = np.clip(y_data_L, 1e-6, 1.0)
y_data_G = np.clip(y_data_G, 1e-6, 1.0)

# 使用 B-Spline 进行平滑拟合
x_smooth = np.linspace(x_data.min(), x_data.max(), 300)

spline_L = make_interp_spline(x_data, y_data_L, k=3)
y_smooth_L = spline_L(x_smooth)
y_smooth_L = np.clip(y_smooth_L, 1e-6, 1.2)

spline_G = make_interp_spline(x_data, y_data_G, k=3)
y_smooth_G = spline_G(x_smooth)
y_smooth_G = np.clip(y_smooth_G, 1e-6, 1.2)

# 绘图配置
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制平滑后的拟合曲线
ax.plot(x_smooth, y_smooth_L, color="#1f77b4", linewidth=2.5, label="Local Cost $C_L$ (Optimizer Driven)", zorder=3)
ax.plot(x_smooth, y_smooth_G, color="#d62728", linewidth=2.5, linestyle="--", label="Global Cost $C_G$ (Passive Tracker)", zorder=3)

# 纵轴核心设置：对数尺度 + 指数格式
ax.set_yscale("log")
ax.set_ylim(1e-5, 1.0)
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
ax.yaxis.set_major_formatter(LogFormatterExponent(base=10.0))

# 细节美化
ax.set_xlabel("Optimization Steps", fontsize=12, fontweight='bold')
ax.set_ylabel("Cost Function (Log Scale)", fontsize=12, fontweight='bold')
ax.set_title("VQLS 1D Bar: Local vs Global Cost", fontsize=15, pad=20, fontweight='bold')

ax.grid(True, which="major", linestyle="-", color='gray', alpha=0.3)
ax.grid(True, which="minor", linestyle="--", color='gray', alpha=0.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)

plt.tight_layout()
plt.show()