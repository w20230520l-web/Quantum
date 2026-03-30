import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import  LogFormatterExponent, LogLocator
from scipy.optimize import minimize


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
wire_map[0] = 0
wire_map[1] = 1
pauli_strings = list()
for op in obs:
    pauli_strings.append(qml.pauli.pauli_word_to_string(op, wire_map=wire_map))

print("--- 刚度矩阵的泡利分解结果 ---")
for i in range(len(c)):
    print(f"项 {i}: 系数 = {c[i]:.2f}, 泡利操作 = {pauli_strings[i]}")

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
# 4. 哈达玛测试(未采用哈达吗重叠测试）
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
    qml.CZ(wires=(ancilla_idx, j))  # 局部控制 Z 门
    U_b()
    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(ancilla_idx))


# ==========================================
# 5. 定义全局与局部代价函数
# ==========================================
def cost_local(weights):
    global_norm = 0.0
    beta_vals = dict()  # 缓存基础内积，防止重复求导计算

    # 步骤1：计算范数分母
    for l in range(len(c)):
        for lp in range(len(c)):
            val = hadamard_test_beta(weights, l, lp)
            beta_vals[(l, lp)] = val
            global_norm += c[l] * c[lp] * val

    # 步骤2：计算局部投影分子
    local_overlap = 0.0
    for l in range(len(c)):
        for lp in range(len(c)):
            z_sum = 0.0
            for j in range(n_qubits):
                z_sum += hadamard_test_local_z(weights, l, lp, j)

            # 数学化简: |0_j><0_j| = 0.5 * (I + Z_j)
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
shape = (n_layers + 1, n_qubits)
w_init = 0.01 * np.random.randn(*shape)
flat_w_init = w_init.flatten()  # SciPy 需要一维数组

history_L = list()
history_G = list()

# 目标函数：供 SciPy 调用，并在内部将一维数组重组为量子电路所需的张量形状
def objective_fn(flat_w):
    w_tensor = np.tensor(flat_w.reshape(shape), requires_grad=True)
    return float(cost_local(w_tensor))

# 梯度函数：利用原论文的 Parameter-Shift 规则获取绝对精确的量子梯度
def gradient_fn(flat_w):
    w_tensor = np.tensor(flat_w.reshape(shape), requires_grad=True)
    grad = qml.grad(cost_local)(w_tensor)
    return grad.flatten()

# 回调函数：用于在绘图前记录局部和全局代价的双轨数据
def callback_fn(flat_w):
    w_tensor = np.tensor(flat_w.reshape(shape), requires_grad=True)
    c_L = cost_local(w_tensor)
    c_G = cost_global(w_tensor)
    history_L.append(float(c_L))
    history_G.append(float(c_G))
    print(f"Step {len(history_L):3d} | Local Cost = {c_L:.5f} | Global Cost = {c_G:.5f}")

print("\n开始 VQLS 量子有限元训练 (使用无学习率参数的 BFGS 优化器)...")
# 记录初始状态
callback_fn(flat_w_init)

# 调用 SciPy 的 BFGS 优化器，彻底免除手动调节学习率
res = minimize(objective_fn,flat_w_init,method='BFGS',jac=gradient_fn,callback=callback_fn,options={'maxiter': 50, 'disp': True})

# 提取最终的最优参数
w_opt = np.tensor(res.x.reshape(shape), requires_grad=False)

# ==========================================
# 7. 结果对比与提取
# ==========================================
dev_state = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev_state)
def get_quantum_state(weights):
    variational_block(weights)
    return qml.state()

quantum_solution = np.real(get_quantum_state(w_opt))

print("\n--- 最终结果对比 ---")
print(f"经典有限元精确解: {classical_normalized}")
print(f"量子变分拟设生成解: {quantum_solution}")

# ==========================================
# 8. 双轨对比平滑绘图 (雪球项目专属高级格式)
# ==========================================
x_data = np.arange(len(history_L))
y_data_L = np.clip(np.array(history_L), 1e-6, 1.0)
y_data_G = np.clip(np.array(history_G), 1e-6, 1.0)

x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
spline_L = make_interp_spline(x_data, y_data_L, k=3)
y_smooth_L = np.clip(spline_L(x_smooth), 1e-6, 1.2)

spline_G = make_interp_spline(x_data, y_data_G, k=3)
y_smooth_G = np.clip(spline_G(x_smooth), 1e-6, 1.2)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_smooth, y_smooth_L, color="#1f77b4", linewidth=2.5, zorder=3)
ax.plot(x_smooth, y_smooth_G, color="#d62728", linewidth=2.5, linestyle="--",  zorder=3)

ax.set_yscale("log")
ax.set_ylim(1e-5, 1.0)
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
ax.yaxis.set_major_formatter(LogFormatterExponent(base=10.0))

ax.set_xlabel("Optimization Steps (BFGS Iterations)", fontsize=12, fontweight='bold')
ax.set_ylabel("Cost Function (Log Scale)", fontsize=12, fontweight='bold')
ax.set_title("Q-FEM 1D Bar: Dynamic Step Size Convergence (Xueqiu Project)", fontsize=15, pad=20, fontweight='bold')

ax.grid(True, which="major", linestyle="-", color='gray', alpha=0.3)
ax.grid(True, which="minor", linestyle="--", color='gray', alpha=0.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)

plt.tight_layout()
plt.show()