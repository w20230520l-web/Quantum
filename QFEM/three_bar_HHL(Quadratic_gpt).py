import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt


# ============================================================
# 1. 三节点二次杆单元：刚度矩阵与局部均布荷载
# ============================================================

def build_three_node_bar_element(E=1.0, A=1.0, L=1.0, q=1.0):
    """
    三节点二次杆单元（单个单元）：
    - 左端节点固定
    - 右端自由
    - 承受均匀局部轴向荷载 q

    返回:
    K_e   : 3x3 单元刚度矩阵
    f_e   : 3x1 单元一致结点力
    K_r   : 2x2 缩减刚度矩阵（保留自由度 u2, u3）
    f_r   : 2x1 缩减载荷
    """
    # 三节点二次杆单元刚度矩阵
    K_e = (E * A / (3.0 * L)) * np.array([
        [ 7.0, -8.0,  1.0],
        [-8.0, 16.0, -8.0],
        [ 1.0, -8.0,  7.0]
    ], dtype=float)

    # 均匀局部荷载的一致结点力
    f_e = (q * L / 6.0) * np.array([1.0, 4.0, 1.0], dtype=float)

    # 左端固定：u1 = 0
    free = [1, 2]  # 对应 u2, u3
    K_r = K_e[np.ix_(free, free)]
    f_r = f_e[free]

    return K_e, f_e, K_r, f_r


# ============================================================
# 2. 经典 FEM 解（用于对照）
# ============================================================

def classical_solve(K_r, f_r):
    return np.linalg.solve(K_r, f_r)


# ============================================================
# 3. HHL 风格量子线路（2x2 系统）
#
# 思想：
#   K_r = U diag(lam0, lam1) U^T
#   在特征基中对 ancilla 做与 1/lam_j 成比例的受控旋转
#   然后后选择 ancilla=1 得到与 K_r^{-1}|b> 成比例的系统态
# ============================================================

def build_hhl_style_circuit(K_r, f_r):
    """
    构建 2x2 SPD 系统的教学型 HHL 风格电路。
    量子位:
      q0 = ancilla
      q1 = system
    """
    # 归一化右端项 |b>
    b_norm = f_r / np.linalg.norm(f_r)

    # 特征分解
    lambdas, U = np.linalg.eigh(K_r)

    # 常数 C，保证 C / lambda_j <= 1
    C = np.min(lambdas)

    # ancilla 旋转角：使 ancilla=1 的幅度为 C/lambda_j
    thetas = 2.0 * np.arcsin(C / lambdas)

    qc = QuantumCircuit(2, 2)

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

    # 测量
    # q0 -> c0 (ancilla)
    # q1 -> c1 (system)
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc, lambdas, U, C


# ============================================================
# 4. 用 shots 运行量子电路并估计：
#    - ancilla=1 成功概率
#    - 条件下系统态分布
#    - 归一化解态 |u>
#    - 物理解向量 u
# ============================================================

def run_hhl_with_shots(qc, K_r, f_r, C, shots=8192, seed=1234):
    simulator = AerSimulator()
    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts(tqc)

    # 计数字符串格式：
    # c1 c0，对应 system ancilla
    #
    # ancilla = 1 的成功事件:
    # "01" -> system=0, anc=1
    # "11" -> system=1, anc=1
    success_01 = counts.get("01", 0)
    success_11 = counts.get("11", 0)
    success_count = success_01 + success_11
    success_prob = success_count / shots

    if success_count == 0:
        raise RuntimeError("ancilla=1 没有采样到，shots 太少或线路设置有问题。")

    # 后选择 ancilla=1 后的系统态概率
    p_sys0_cond = success_01 / success_count
    p_sys1_cond = success_11 / success_count

    # 由于本例中位移分量均为正，可直接取振幅正根
    u_state_normed = np.array([
        np.sqrt(p_sys0_cond),
        np.sqrt(p_sys1_cond)
    ], dtype=float)

    # HHL 输出的是归一化方向，需要恢复物理尺度
    # 对归一化输入 b_norm = f_r / ||f_r||，有：
    #   success_prob = C^2 * ||K^{-1} b_norm||^2
    # 所以
    #   ||u_r|| = ||f_r|| * ||K^{-1} b_norm|| = ||f_r|| * sqrt(p_success) / C
    u_norm_physical = np.linalg.norm(f_r) * np.sqrt(success_prob) / C

    # 恢复物理解
    u_quantum = u_norm_physical * u_state_normed

    return {
        "counts": counts,
        "success_prob": success_prob,
        "u_state_normed": u_state_normed,
        "u_norm_physical": u_norm_physical,
        "u_quantum": u_quantum,
        "p_sys0_cond": p_sys0_cond,
        "p_sys1_cond": p_sys1_cond,
    }


# ============================================================
# 5. 泛函：自由端节点应变能贡献
#
# 定义:
#   J(u) = 1/2 * f_free * u_free
#
# 这里 reduced vector = [u2, u3]
# 因此自由端节点是第 2 个分量 (索引 1)
#
# 用“泛函思想”表示:
#   J(u) = r^T u
# 其中
#   r = [0, 1/2 * f3]^T
#
# 因本例 r 仅在第二分量非零，所以 |r> = |1>
# 则：
#   J(u) = ||r|| * ||u|| * <r|u_norm>
#        = (1/2*f3) * ||u|| * amplitude_on_|1>
# ============================================================

def free_node_strain_energy_functional(f_r, u_quantum, u_state_normed, u_norm_physical):
    # 自由端节点对应 reduced vector 的第二个分量
    f_free = f_r[1]
    amp_free = u_state_normed[1]   # |1> 振幅（正实）
    r_norm = 0.5 * f_free

    # 泛函形式估计
    J_quantum = r_norm * u_norm_physical * amp_free

    # 直接由恢复后的物理解计算（应当与上面一致）
    J_check = 0.5 * f_free * u_quantum[1]

    return J_quantum, J_check


# ============================================================
# 6. 也给出总应变能，便于比较
# ============================================================

def total_strain_energy(K_r, u):
    return 0.5 * float(u.T @ K_r @ u)


# ============================================================
# 7. 主程序
# ============================================================

if __name__ == "__main__":
    # ---------------- 参数 ----------------
    E = 1.0
    A = 1.0
    L = 1.0
    q = 1.0
    shots = 8192

    # ---------------- 建模 ----------------
    K_e, f_e, K_r, f_r = build_three_node_bar_element(E, A, L, q)

    # 经典解
    u_classical = classical_solve(K_r, f_r)

    # 经典泛函（自由端节点应变能贡献）
    J_classical = 0.5 * f_r[1] * u_classical[1]

    # 经典总应变能
    U_total_classical = total_strain_energy(K_r, u_classical)

    # ---------------- 量子 HHL 风格 ----------------
    qc, lambdas, U, C = build_hhl_style_circuit(K_r, f_r)
    qres = run_hhl_with_shots(qc, K_r, f_r, C, shots=shots, seed=1234)

    u_quantum = qres["u_quantum"]
    u_state_normed = qres["u_state_normed"]
    u_norm_physical = qres["u_norm_physical"]

    # 泛函：自由端节点应变能贡献
    J_quantum, J_check = free_node_strain_energy_functional(
        f_r, u_quantum, u_state_normed, u_norm_physical
    )

    # 总应变能
    U_total_quantum = total_strain_energy(K_r, u_quantum)

    # ---------------- 输出 ----------------
    print("========== 三节点二次杆单元 ==========")
    print("单元刚度矩阵 K_e =\n", K_e)
    print("一致结点力 f_e =", f_e)
    print("\n缩减刚度矩阵 K_r =\n", K_r)
    print("缩减载荷 f_r =", f_r)

    print("\n========== 经典 FEM 结果 ==========")
    print("经典自由度解 u_r =", u_classical)
    print("经典自由端节点应变能贡献 J = 1/2 f3 u3 =", J_classical)
    print("经典总应变能 U = 1/2 u^T K u =", U_total_classical)

    print("\n========== 量子 HHL 风格 + shots ==========")
    print("特征值 lambdas =", lambdas)
    print("C =", C)
    print("counts =", qres["counts"])
    print("ancilla=1 成功概率 =", qres["success_prob"])
    print("后选择后归一化解态 |u> =", u_state_normed)
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
    # 使用 output='mpl' 来生成 matplotlib 图像
    qc.draw(output='mpl', fold=-1)

    # 展示图像
    plt.show()