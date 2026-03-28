import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt


# ============================================================
# 1. 经典 FEM：两单元一维杆，左端固定，右端单位力
# ============================================================

def build_two_bar_reduced_system(E=1.0, A=1.0, L=1.0, F=1.0):
    """
    两单元串联杆，节点1固定，节点3受力F。
    每个单元长度都取 L，材料参数 E, A 一致。

    返回:
    K_full : 3x3 总刚度矩阵
    f_full : 3x1 总载荷向量
    K_r    : 2x2 缩减刚度矩阵（保留自由度 u2, u3）
    f_r    : 2x1 缩减载荷
    """
    ke = (E * A / L) * np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ], dtype=float)

    # 总刚度矩阵
    K = np.zeros((3, 3), dtype=float)

    # 单元1: 节点(1,2)
    dofs_e1 = [0, 1]
    for i in range(2):
        for j in range(2):
            K[dofs_e1[i], dofs_e1[j]] += ke[i, j]

    # 单元2: 节点(2,3)
    dofs_e2 = [1, 2]
    for i in range(2):
        for j in range(2):
            K[dofs_e2[i], dofs_e2[j]] += ke[i, j]

    # 总载荷
    f = np.array([0.0, 0.0, F], dtype=float)

    # 左端固定: u1 = 0
    free = [1, 2]   # 对应 u2, u3
    K_r = K[np.ix_(free, free)]
    f_r = f[free]

    return K, f, K_r, f_r


# ============================================================
# 2. 经典解
# ============================================================

def classical_solve(K_r, f_r):
    return np.linalg.solve(K_r, f_r)


# ============================================================
# 3. 教学型 HHL 风格求解器（2x2 SPD 系统）
#
# 思想：
#   K_r = U diag(lam0, lam1) U^T
#   在特征基中对 ancilla 做与 1/lam_j 成比例的受控旋转
#   然后后选择 ancilla=1 得到与 K_r^{-1}|b> 成比例的系统态
# ============================================================

def hhl_style_two_by_two(K_r, f_r):
    """
    返回:
    qc               : 构造好的量子线路
    lambdas          : K_r 的特征值
    U                : 特征向量矩阵
    C                : 受控旋转缩放常数
    x_quantum_normed : 后选择后的归一化解态
    success_prob     : ancilla=1 的成功概率
    scale_alpha      : 恢复物理尺度因子
    x_quantum_phys   : 恢复物理尺度后的量子解
    """
    # 归一化右端项 |b>
    b_norm = f_r / np.linalg.norm(f_r)

    # 特征分解
    lambdas, U = np.linalg.eigh(K_r)

    # 常数 C，保证 C / lambda_j <= 1
    C = np.min(lambdas)

    # ancilla 旋转角
    thetas = 2.0 * np.arcsin(C / lambdas)

    # q0 = ancilla, q1 = system
    qc = QuantumCircuit(2)

    # (a) 准备 |b> 到 system qubit (q1)
    qc.initialize(b_norm, 1)

    # (b) 变到特征基：U^\dagger
    qc.unitary(Operator(U.conj().T), [1], label="U_dag")

    # (c) 对不同特征态做受控旋转
    # 特征态 |0> 分支：control on |0>
    qc.x(1)
    qc.cry(thetas[0], 1, 0)
    qc.x(1)

    # 特征态 |1> 分支：control on |1>
    qc.cry(thetas[1], 1, 0)

    # (d) 回到原基
    qc.unitary(Operator(U), [1], label="U")

    # 状态向量模拟
    sv = Statevector.from_instruction(qc)
    amps = sv.data

    # Qiskit 两比特索引:
    # index 0 -> |00>
    # index 1 -> |01>
    # index 2 -> |10>
    # index 3 -> |11>
    #
    # 其中 q0=ancilla, q1=system
    # ancilla=1 对应索引 1, 3
    post = np.array([amps[1], amps[3]], dtype=complex)
    success_prob = np.sum(np.abs(post) ** 2)

    if success_prob < 1e-14:
        raise RuntimeError("ancilla=1 success probability is too small.")

    # 后选择后的归一化解态
    x_quantum_normed = post / np.sqrt(success_prob)
    x_quantum_normed = np.real_if_close(x_quantum_normed)

    # --------------------------------------------------------
    # 恢复物理尺度
    # 用残差最小化求 alpha:
    #   alpha = argmin ||K(alpha x_q) - f||
    #   alpha = (Kx · f) / ||Kx||^2
    # --------------------------------------------------------
    Kx = K_r @ x_quantum_normed
    scale_alpha = float(np.dot(Kx, f_r) / np.dot(Kx, Kx))
    x_quantum_phys = scale_alpha * x_quantum_normed

    return qc, lambdas, U, C, x_quantum_normed, success_prob, scale_alpha, x_quantum_phys


# ============================================================
# 4. 泛函：自由端节点应变能贡献
#
# 定义:
#   J(u) = 1/2 * f_free * u_free
#
# reduced vector = [u2, u3]
# 因此自由端节点是第 2 个分量 (索引 1)
#
# 泛函表示:
#   J(u) = r^T u
# 其中
#   r = [0, 1/2 * f3]^T
# ============================================================

def free_node_strain_energy_functional(f_r, u_quantum, u_state_normed, u_norm_physical):
    f_free = f_r[1]
    amp_free = u_state_normed[1]
    r_norm = 0.5 * f_free

    # 泛函形式估计
    J_quantum = r_norm * u_norm_physical * amp_free

    # 直接用恢复后的物理解验证
    J_check = 0.5 * f_free * u_quantum[1]

    return J_quantum, J_check


# ============================================================
# 5. 总应变能
# ============================================================

def total_strain_energy(K_r, u):
    return 0.5 * float(u.T @ K_r @ u)


# ============================================================
# 6. 主程序
# ============================================================

if __name__ == "__main__":
    # ---------------- 参数 ----------------
    E = 1.0
    A = 1.0
    L = 1.0
    F = 1.0

    # ---------------- 建模 ----------------
    K_full, f_full, K_r, f_r = build_two_bar_reduced_system(E, A, L, F)

    # 经典解
    u_classical = classical_solve(K_r, f_r)

    # 经典泛函（自由端节点应变能贡献）
    J_classical = 0.5 * f_r[1] * u_classical[1]

    # 经典总应变能
    U_total_classical = total_strain_energy(K_r, u_classical)

    # ---------------- 量子 HHL 风格 ----------------
    qc, lambdas, U, C, u_state_normed, success_prob, alpha, u_quantum = hhl_style_two_by_two(K_r, f_r)

    # 这里恢复尺度后的模长
    u_norm_physical = np.linalg.norm(u_quantum)

    # 泛函：自由端节点应变能贡献
    J_quantum, J_check = free_node_strain_energy_functional(
        f_r, u_quantum, u_state_normed, u_norm_physical
    )

    # 总应变能
    U_total_quantum = total_strain_energy(K_r, u_quantum)

    # ---------------- 输出 ----------------
    print("========== 两单元一维杆单元 ==========")
    print("总刚度矩阵 K =\n", K_full)
    print("总载荷 f =", f_full)

    print("\n缩减刚度矩阵 K_r =\n", K_r)
    print("缩减载荷 f_r =", f_r)

    print("\n========== 经典 FEM 结果 ==========")
    print("经典自由度解 u_r =", u_classical)
    print("经典自由端节点应变能贡献 J = 1/2 f3 u3 =", J_classical)
    print("经典总应变能 U = 1/2 u^T K u =", U_total_classical)

    print("\n========== 量子 HHL 风格 + 状态向量 ==========")
    print("特征值 lambdas =", lambdas)
    print("C =", C)
    print("ancilla=1 成功概率 =", success_prob)
    print("后选择后归一化解态 |u> =", u_state_normed)
    print("恢复物理尺度 alpha =", alpha)
    print("恢复的物理解 u_r^(quantum) =", u_quantum)

    print("\n========== 泛函输出 ==========")
    print("量子泛函估计 J_quantum =", J_quantum)
    print("量子直接验证 J_check   =", J_check)
    print("经典泛函 J_classical   =", J_classical)

    print("\n========== 总应变能对比 ==========")
    print("量子总应变能 U_total_quantum =", U_total_quantum)
    print("经典总应变能 U_total_classical =", U_total_classical)

    print("\n========== 误差 ==========")
    abs_err_u = np.linalg.norm(u_quantum - u_classical)
    rel_err_u = abs_err_u / np.linalg.norm(u_classical)

    abs_err_J = abs(J_quantum - J_classical)
    rel_err_J = abs_err_J / abs(J_classical)

    print("位移绝对误差 ||u_q - u_c|| =", abs_err_u)
    print("位移相对误差 =", rel_err_u)
    print("泛函绝对误差 |J_q - J_c| =", abs_err_J)
    print("泛函相对误差 =", rel_err_J)

    print("\n=== 量子线路 ===")
    qc.draw(output='mpl', fold=-1)
    plt.show()