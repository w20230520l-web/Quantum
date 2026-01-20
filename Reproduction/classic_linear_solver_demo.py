import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, state_fidelity
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt
from qiskit.circuit.library import CUGate
from qiskit.circuit.library import UnitaryGate

# 1. 初始制备
def get_problem():
    A_raw = np.array([[3.0, 1.0],
                      [1.0, 3.0]])
    # 归一化
    s = 4.0
    A = A_raw / s

    b_vec = np.array([1.0, 0.0])
    # 转化为量子态对象
    psi_b = Statevector(b_vec)
    return A, psi_b, s

# 2. 四项算符项
def calculate_gradient_terms(rho_current, x_vec_current, A, psi_b):
    I = np.eye(2)
    At = A.conj().T
    AtA = At @ A

    # 当前的向量 x (为了计算交叉项，我们需要知道当前的相位信息)
    # 在纯量子算法中，这通过 Block Encoding 的乘法实现
    x = x_vec_current
    magic_scale = 2.52982212
    b = (psi_b.data / 4.0) * magic_scale
    grad_vec = AtA @ x - (At @ b)

    # 1. 第一项: x * x^† (原始密度矩阵)
    term_1 = np.outer(x, x.conj())
    # 2. 第二项: -η * x * ∇f^† (左交叉项)
    term_2 = np.outer(x, grad_vec.conj())
    # 3. 第三项: -η * ∇f * x^† (右交叉项，是第二项的共轭转置)
    term_3 = np.outer(grad_vec, x.conj())
    # 4. 第四项: +η^2 * ∇f * ∇f^† (二次修正项)
    term_4 = np.outer(grad_vec, grad_vec.conj())
    return term_1, term_2, term_3, term_4, grad_vec

# 3. 运行迭代
def run_density_matrix_solver():
    A, psi_b, s = get_problem()
    x_current = np.array([0.6, 0.8])
    rho_current = DensityMatrix(Statevector(x_current))  # 将 x0 转化为密度矩阵

    eta = 0.8  # 学习率
    iterations = 15  # 迭代次数

    fidelities = []

    # 计算真实解
    # 解 A * x = b (注意这里的 A 是归一化后的)
    x_true_raw = np.linalg.inv(A) @ psi_b.data
    x_true = x_true_raw / np.linalg.norm(x_true_raw)
    rho_true = DensityMatrix(Statevector(x_true))

    print(f"目标: 求解 Ax = b")
    print(f"真实解向量 (方向): {np.round(x_true, 4)}")
    print("-" * 65)

    # --- 打印表头 (Header) ---
    # {:<5} 表示左对齐占5格，方便看数据
    print(f"{'Iter':<5} | {'Fidelity':<10} | {'Current Vector (x)':<30}")
    print("-" * 65)
    initial_fid = state_fidelity(rho_current, rho_true)
    print(f"{'Init':<5} | {initial_fid:.6f}   | {str(np.round(x_current, 4))}")
    for t in range(iterations):
        # 计算展开的四项
        T1, T2, T3, T4, grad_vec = calculate_gradient_terms(rho_current, x_current, A, psi_b)
        rho_matrix_new = T1 - eta * T2 - eta * T3 + (eta ** 2) * T4

        # 更新向量
        x_new = x_current - eta * grad_vec

        #  归一化 (两个都要)
        trace_val = np.trace(rho_matrix_new)
        if abs(trace_val) > 1e-9:
            rho_matrix_new /= trace_val

        if np.linalg.norm(x_new) > 1e-9:
            x_new /= np.linalg.norm(x_new)

        # 更新当前状态
        rho_current = DensityMatrix(rho_matrix_new)
        x_current = x_new  # 仅用于生成下一轮的交叉项，实际物理对象是 rho

        # 计算保真度 (Fidelity) - 衡量当前密度矩阵和真实解的密度矩阵有多像
        # ✅ 使用 state_fidelity 函数
        fid = state_fidelity(rho_current, rho_true)
        fidelities.append(fid)

        vec_str = str(np.round(x_current, 4))  # 把向量转成字符串，保留4位小数
        print(f"{t:<5} | {fid:.6f}   | {vec_str}")

    # ==========================================
    # 4. 可视化结果
    # ==========================================
    print("-" * 65)
    print(f"最终保真度: {fidelities[-1]:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(fidelities, 'o-', color='purple', markersize=5)
    plt.axhline(1.0, color='r', linestyle='--', label='Target')
    plt.title("Convergence Process (All Steps)")
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 1. 运行你的核心密度矩阵求解器
    run_density_matrix_solver()