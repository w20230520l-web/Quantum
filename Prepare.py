from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.circuit.library import StatePreparation
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import numpy as np

def prepare_quantum_state(*,
                            amplitudes=None,
                            probs=None,
                            phases=None,
                            add_measure=True):
    """
    传入幅度向量 amplitudes（complex，长度=2^n），
    或者 probs(实数非负，和为1) + phases(与 probs 同长度，单位弧度)。
    返回：量子线路 qc（从 |0...0> 制备到 |psi>）。
    """

    # 第一步：组装幅度向量
    if amplitudes is None:
        if probs is None:
            raise ValueError("请提供 amplitudes，或 probs(+可选 phases)。")
        probs = np.asarray(probs, dtype=float).flatten()  # 转为一维数组
        if (probs < -1e-12).any():
            raise ValueError("probs 含负数。")
        s = probs.sum()  # 求和
        if s <= 0:
            raise ValueError("probs 全为0。")
        probs = probs / s  # 归一化
        if phases is None:
            phases = np.zeros_like(probs, dtype=float)  # 默认全0相位
        phases = np.asarray(phases, dtype=float).flatten()
        if len(phases) != len(probs):
            raise ValueError("probs 与 phases 长度不一致。")
        amplitudes = np.sqrt(probs) * np.exp(1j * phases)
    else:
        amplitudes = np.asarray(amplitudes, dtype=complex).flatten()

    # 第二步：维度检查与归一化
    L = len(amplitudes)
    if L == 0 or (L & (L - 1)) != 0:
        raise ValueError("幅度向量长度必须是 2^n。当前长度 = %d" % L)
    n = int(np.log2(L))
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("幅度全为0。")
    amplitudes = amplitudes / norm  # 归一化

    # 第三步：生成电路
    qc = QuantumCircuit(n, n if add_measure else 0)
    sp = StatePreparation(amplitudes)  # 自动分解到基础门
    qc.append(sp, qc.qubits)

    if add_measure:
        qc.measure(range(n), range(n))  # q[i] -> c[i]（小端序）
    return qc


# ====== 使用示例 ======
if __name__ == "__main__":
    # 例1：直接给幅度（2比特，长度4）
    amps = np.array([0.4, 0.4, 0.8, 0.2], dtype=complex)  # (|00> + |11>)/√2
    qc = prepare_quantum_state(amplitudes=amps, add_measure=True)
    fig1 = qc.draw(output='mpl', fold=120)
    plt.show()

    # 例2：给概率+相位（2比特，长度4）
    probs = [0.16, 0.16, 0.64, 0.04]  # 和会被自动归一化
    phases = [0, 0, 0, 0]
    qc3 = prepare_quantum_state(probs=probs, phases=phases, add_measure=True)
    fig1 = qc3.draw(output='mpl', fold=120)
    plt.show()

    # 计数直方图对比
    backend = Aer.get_backend("aer_simulator")
    job1 = backend.run(transpile(qc, backend), shots=20000)
    job2 = backend.run(transpile(qc3, backend), shots=20000)
    c1 = job1.result().get_counts()
    c2 = job2.result().get_counts()
    fig = plot_histogram([c1, c2], legend=["Case 1 (amps)", "Case 2 (probs+phases)"],
                         title="Counts comparison")
    plt.show()


    # 模拟验证
    backend = Aer.get_backend("aer_simulator")
    for circuit in [qc, qc3]:
        job = backend.run(transpile(circuit, backend), shots=20000)
        counts = job.result().get_counts()
        print(counts)