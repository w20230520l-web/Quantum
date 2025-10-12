from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import StatePreparation
from qiskit_aer import Aer
import numpy as np
import matplotlib.pyplot as plt
##

def prepare_quantum_state(*,
                          amplitudes = None,
                          probs = None,
                          phases = None,
                          add_measures = True):
    """
    传入：
    amplitudes: 复数幅度向量，长度为2^n
    probs: 实数概率向量，长度为2^n，非负且和为1
    phsaes: 实数相位向量，长度为2^n，单位弧度
    add_measures: 是否添加测量操作，默认添加
    返回：
    量子线路 qc（从 |0...0> 制备到 |psi>）。
    """

    #第一步： 组装幅度向量
    if amplitudes is None:
        if probs is None:
            raise ValueError ("请提供 amplitudes， 或 probs(+phases)。")
        probs = np.asarray(probs, dtype = float).flatten()  # 转为一维数组
        if (probs < -1e-12).any():
            raise ValueError("probs 含负数")
        s=probs.sum()
        if  s <= 0:
            raise ValueError("probs 全为0")
        probs = probs/s    # 归一化
        if phases is None:
            phases = np.zeros_like(probs, dtype = float)   # 默认全0相位
        phases = np.asarray(phases, dtype = float).flatten()   # 转为一维数组
        if len(phases) != len(probs):
            raise ValueError("phases 与 probs 维度不同")
        amplitudes = np.sqrt(probs) * np.exp(1j * phases)   # 复数幅度向量
    else:
        amplitudes = np.asarray(amplitudes,dtype=complex).flatten()

    # 第二步：维度检查与归一化
    L = len(amplitudes)
    if L == 0 or (L&(L-1)) != 0:
        raise ValueError("幅度向量必须是 2^n")
    n = int(np.log2(L))
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("幅度全为0")
    amplitudes = amplitudes / norm


    # 第三步：生成电路
    qc = QuantumCircuit(n, n if add_measures else 0)
    sp = StatePreparation(amplitudes)   # 自动分解到基础门
    qc.append(sp, qc.qubits)

    if add_measures:
        qc.measure(range(n), range(n))   # q[i] -> c[i]（小端序）
    return qc


# 例子：
if __name__ == "__main__":
    # 例1：
    amps = np.array([0.4, 0.4, 0.8, 0.2], dtype=complex)
    qc1 = prepare_quantum_state(amplitudes = amps, add_measures = True)
    fig1 = qc1.draw(output = 'mpl', fold = 120)
    plt.show()

    #  例2：
    probs = np.array([0.16,0.16,0.64,0.04])
    phases = np.array([0, 0, 0, 0])
    qc2 = prepare_quantum_state(probs = probs, phases = phases, add_measures = True)
    fig2 = qc2.draw(output = 'mpl', fold = 120)
    plt.show()


    # 直方图对比：
    backend = Aer.get_backend("aer_simulator")
    rest1 = backend.run(transpile(qc1, backend),shots = 20000)
    rest2 = backend.run(transpile(qc2, backend),shots = 20000)
    result1 = rest1.result().get_counts()
    result2 = rest2.result().get_counts()
    fig3 = plot_histogram([result1, result2], legend = ["Case1 (amps)", "Case2 (probs+phases)"],
                          title = "Counts comparison")
    plt.show()


   # 模拟验证
    backend = Aer.get_backend("aer_simulator")
    for circuit in [qc1, qc2]:
        job = backend.run(transpile(circuit, backend), shots=20000)
        counts = job.result().get_counts()
        print(counts)





