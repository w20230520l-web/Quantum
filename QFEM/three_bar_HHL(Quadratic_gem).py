import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt


# ============================================================
# 1. 高阶 FEM：单单元三节点一维杆，均布荷载，左固定，右自由
# ============================================================
def build_three_node_bar(E=1.0, A=1.0, L=1.0, q=1.0):
    """
    节点1: x=0 (固定), 节点2: x=L/2, 节点3: x=L (自由)
    均布荷载 q = 6.0
    """
    K = (E * A / (3.0 * L)) * np.array([
        [7.0, -8.0, 1.0],
        [-8.0, 16.0, -8.0],
        [1.0, -8.0, 7.0]
    ])
    f = (q * L / 6.0) * np.array([1.0, 4.0, 1.0])
    free = [1, 2]
    K_r = K[np.ix_(free, free)]
    f_r = f[free]
    return K_r, f_r, K, f


def classical_solve_and_energy(K_r, f_r):
    u_r = np.linalg.solve(K_r, f_r)
    strain_energy = 0.5 * np.dot(u_r, np.dot(K_r, u_r))
    return u_r, strain_energy


# ============================================================
# 2. 纯理论型 HHL 求解器 (公式法恢复物理尺度，带 Shot 采样)
# ============================================================
def hhl_with_shots_theoretical(K_r, f_r, shots=8192):
    norm_f = np.linalg.norm(f_r)
    b_norm = f_r / norm_f

    lambdas, U = np.linalg.eigh(K_r)
    C = np.min(lambdas)
    thetas = 2.0 * np.arcsin(C / lambdas)

    qc = QuantumCircuit(2, 2)
    qc.initialize(b_norm, 1)
    qc.unitary(Operator(U.conj().T), [1], label="U_dag")

    qc.x(1)
    qc.cry(thetas[0], 1, 0)
    qc.x(1)
    qc.cry(thetas[1], 1, 0)

    qc.unitary(Operator(U), [1], label="U")
    qc.measure([0, 1], [0, 1])

    # 执行采样
    simulator = AerSimulator()
    compiled_qc = transpile(qc, simulator)
    job = simulator.run(compiled_qc, shots=shots)
    counts = job.result().get_counts()

    # 读取字典键值：Qiskit 默认输出格式为无空格字符串 'c1c0'
    count_01 = counts.get('01', 0)  # ancilla=1, system=0
    count_11 = counts.get('11', 0)  # ancilla=1, system=1

    success_shots = count_01 + count_11
    if success_shots == 0:
        raise RuntimeError("零次成功采样，请检查线路或增加 shots 数量。")

    p_success = success_shots / shots

    # 统计振幅
    prob_0 = count_01 / success_shots
    prob_1 = count_11 / success_shots
    x_q_normed_sampled = np.array([np.sqrt(prob_0), np.sqrt(prob_1)])

    # ==========================================
    # 核心：纯理论公式法恢复物理尺度
    # x = ( ||f|| * sqrt(P_success) / C ) * x_normed
    # ==========================================
    theoretical_scale = (norm_f * np.sqrt(p_success)) / C
    x_q_physical = theoretical_scale * x_q_normed_sampled

    # 量子求解下的系统应变能 U = 1/2 * u^T * K * u
    q_strain_energy = 0.5 * np.dot(x_q_physical, np.dot(K_r, x_q_physical))

    return qc, x_q_normed_sampled, p_success, x_q_physical, q_strain_energy, counts


# ============================================================
# 3. 主程序
# ============================================================
if __name__ == "__main__":
    K_r, f_r, K_full, f_full = build_three_node_bar()
    u_classical, classical_energy = classical_solve_and_energy(K_r, f_r)

    SHOTS = 10000
    qc, x_q_normed, p_success, x_q_physical, q_energy, raw_counts = hhl_with_shots_theoretical(K_r, f_r, shots=SHOTS)

    print("=== 三节点有限元模型 (经典解) ===")
    print("缩减刚度矩阵 K_r =\n", np.round(K_r, 4))
    print("等效节点力 f_r =", f_r)
    print("经典节点位移 [u2, u3] =", np.round(u_classical, 4))
    print(f"系统总应变能 U = {classical_energy:.4f}")

    print(f"\n=== 量子 HHL 采样结果 (纯理论公式法, Shots = {SHOTS}) ===")
    print("原始测量频次 (格式 'q1q0') =", raw_counts)
    print("Ancilla=|1> 成功概率 ≈ {:.2f}%".format(p_success * 100))
    print("量子估算归一化态 =", np.round(x_q_normed, 4))
    print("量子还原物理位移 =", np.round(x_q_physical, 4))
    print(f"量子计算得出的系统应变能 = {q_energy:.4f}")

    print("\n=== 对比与验证 ===")
    err = np.linalg.norm(x_q_physical - u_classical)
    rel_err = err / np.linalg.norm(u_classical)
    print("位移绝对误差 =", err)
    print("位移相对误差 =", rel_err)
    print("应变能相对误差 =", abs(q_energy - classical_energy) / classical_energy)

    # 绘制线路图
    qc.draw(output='mpl', fold=-1)
    plt.show()