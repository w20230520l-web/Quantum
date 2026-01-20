import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, state_fidelity
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt

# ==========================================
# 1. 准备环境 (Setup)
# ==========================================

def get_problem():
    # 为了演示，我们使用 2x2 矩阵
    # 原始 A
    A_raw = np.array([[3.0, 1.0],
                      [1.0, 3.0]])
    # 归一化 (满足 ||A|| <= 1)
    s = 4.0
    A = A_raw / s

    # 目标态 |b> (单位向量)
    b_vec = np.array([1.0, 0.0])
    # 转化为量子态对象
    psi_b = Statevector(b_vec)

    return A, psi_b, s


# ==========================================
# 2. 核心：构造论文中的算符项
# ==========================================

def calculate_gradient_terms(rho_current, x_vec_current, A, psi_b):
    """
    这里我们不直接算 x_new，而是分别算出论文 Eq.13 中的四项矩阵
    来验证 '密度算符更新' 的逻辑。
    """
    # 辅助变量
    I = np.eye(2)
    At = A.conj().T
    AtA = At @ A

    # 当前的向量 x (为了计算交叉项，我们需要知道当前的相位信息)
    # 在纯量子算法中，这通过 Block Encoding 的乘法实现
    x = x_vec_current
    b = psi_b.data

    # --- 构造梯度算子 ∇f ---
    # ∇f(x) = (I + A^† A)x - A^† b  [cite: 160]
    grad_vec = (I + AtA) @ x - (At @ b)

    # --- 构造四项展开式 (对应论文 Eq. 167 - Eq. 170) ---

    # 1. 第一项: xx^† (旧的密度矩阵)
    # [cite: 169] "xx^†"
    term_1 = np.outer(x, x.conj())

    # 2. 第二项: -η * x * ∇f^† (左交叉项)
    # [cite: 171] "x((I + A^† A)x - A^†|b>)^†"
    # 这一项是 x 乘以 梯度的共轭转置
    term_2 = np.outer(x, grad_vec.conj())

    # 3. 第三项: -η * ∇f * x^† (右交叉项，是第二项的共轭转置)
    # [cite: 169] "((I + A^† A)x - A^†|b>)x^†"
    term_3 = np.outer(grad_vec, x.conj())

    # 4. 第四项: +η^2 * ∇f * ∇f^† (二次修正项)
    # [cite: 170]
    term_4 = np.outer(grad_vec, grad_vec.conj())

    return term_1, term_2, term_3, term_4, grad_vec


# ==========================================
# 3. 运行迭代 (Density Matrix Descent)
# ==========================================

def run_density_matrix_solver():
    A, psi_b, s = get_problem()

    # 初始化：从 |b> 开始猜
    # x0 = b
    x_current = psi_b.data
    rho_current = DensityMatrix(psi_b)  # 将 x0 转化为密度矩阵

    # 超参数
    eta = 0.5  # 学习率
    iterations = 20

    fidelities = []

    print(f"目标: 求解 Ax = b (在密度矩阵空间中迭代)")
    print(f"初始保真度 (Fidelity): {1.0:.4f} (因为 x0=b)")
    print("-" * 40)

    # 计算真实解 (用于验证)
    # 解 A * x = b (注意这里的 A 是归一化后的)
    x_true = np.linalg.inv(A) @ psi_b.data
    # 归一化真实解，因为量子态只看方向
    x_true = x_true / np.linalg.norm(x_true)
    rho_true = DensityMatrix(Statevector(x_true))

    for t in range(iterations):
        # 计算展开的四项
        T1, T2, T3, T4, grad_vec = calculate_gradient_terms(rho_current, x_current, A, psi_b)

        # === 核心：密度矩阵更新公式  ===
        # rho_new = rho_old - eta * (Left_Cross) - eta * (Right_Cross) + eta^2 * (Quadratic)
        # 注意：这里是矩阵加减法！不是向量加减法！
        rho_matrix_new = T1 - eta * T2 - eta * T3 + (eta ** 2) * T4

        # 更新向量 (用于下一轮计算辅助)
        # x_new = x_old - eta * grad
        x_new = x_current - eta * grad_vec

        # 归一化密度矩阵 (量子态必须归一化)
        # 在论文算法中，这一步通过测量后的 Post-selection (后选择) 概率归一化实现
        trace_val = np.trace(rho_matrix_new)
        rho_matrix_new /= trace_val

        # 更新当前状态
        rho_current = DensityMatrix(rho_matrix_new)
        x_current = x_new  # 仅用于生成下一轮的交叉项，实际物理对象是 rho

        # 计算保真度 (Fidelity) - 衡量当前密度矩阵和真实解的密度矩阵有多像
        # ✅ 使用 state_fidelity 函数
        fid = state_fidelity(rho_current, rho_true)
        fidelities.append(fid)

        if t % 5 == 0:
            print(f"Iter {t}: Fidelity = {fid:.6f}")

    # ==========================================
    # 4. 可视化结果
    # ==========================================
    print("-" * 40)
    print(f"最终保真度: {fidelities[-1]:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(fidelities, marker='o', linestyle='-', color='purple')
    plt.axhline(1.0, color='r', linestyle='--', label='Perfect Match')
    plt.title("Convergence of Density Matrix Iteration (Fidelity)")
    plt.xlabel("Iterations (t)")
    plt.ylabel("Fidelity with True Solution")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 打印最终的矩阵看看
    print("\n最终求解得到的密度矩阵 rho_final:")
    print(np.round(rho_current.data, 3))
    print("\n真实解的密度矩阵 rho_true:")
    print(np.round(rho_true.data, 3))


def  visualize_hardware_implementation():
    print("\n" + "=" * 40)
    print("🔍 [微观视角] 正在'拆包'迭代线路，展示内部的基础门...")
    print("=" * 40)

    # --- 1. 定义基础积木 (这次我们放点真实的门进去) ---

    # 模拟 x0 制备: H 门 + T 门
    def make_U_x0():
        qc = QuantumCircuit(1, name="Init_x0")
        qc.h(0)
        qc.t(0)
        return qc.to_gate()  # 打包

    # 模拟 梯度算子: X 门 + Rz 旋转
    def make_Gradient_Op():
        qc = QuantumCircuit(1, name="Grad_Op")
        qc.x(0)
        qc.rz(0.5, 0)
        return qc.to_gate()  # 打包

    # --- 2. 构建迭代线路 (打包状态) ---

    # 第一次迭代
    qc_iter1 = QuantumCircuit(1, name="Iter1")
    qc_iter1.append(make_U_x0(), [0])
    qc_iter1.append(make_Gradient_Op(), [0])

    # 第二次迭代 (套娃)
    qc_iter2 = QuantumCircuit(1, name="Iter2")
    prev_gate = qc_iter1.to_gate()
    prev_gate.name = "Previous_Iter"

    qc_iter2.append(prev_gate, [0])  # 放入上一次迭代的大盒子
    qc_iter2.append(make_Gradient_Op(), [0])  # 放入这次的梯度

    # --- 3. 对比展示 ---

    # [图 1] 打包版 (你之前看到的)
    print("   -> 图 1: 宏观视角 (盒子)")
    qc_iter2.draw('mpl', style='iqp')
    plt.title("Fig 1: High-level (Black Boxes)")
    plt.show()

    # [图 2] 拆包版 (你要的门细节！)
    # decompose() 只能拆一层，transpile 可以彻底拆到底
    print("   -> 图 2: 微观视角 (炸开盒子看细节)")

    # 使用 transpile 将所有盒子炸开，分解成基础门 (u, cx, rz...)
    qc_decomposed = transpile(qc_iter2, basis_gates=['h', 'x', 'rz', 't'], optimization_level=0)

    qc_decomposed.draw('mpl', style='iqp')
    plt.title("Fig 2: Low-level (Real Gates Revealed)")
    plt.show()
if __name__ == "__main__":
    # 1. 运行你的核心密度矩阵求解器
    run_density_matrix_solver()

    # 2. 运行新增的可视化模块
    visualize_hardware_implementation()