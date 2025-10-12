from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_distribution
import numpy as np
import matplotlib.pyplot as plt

def _bitstr_to_little_endian_bits(bstr: str, n: int):
    """把高位在左的比特串（如 '101'）转成按 q0 最右的小端顺序列表。"""
    b = bstr.strip().replace(" ", "")
    if len(b) != n:
        raise ValueError(f"bitstring 长度应为 {n}，得到 {len(b)}: {bstr!r}")
    return [int(ch) for ch in b[::-1]]  # q0 对应末位

def _phase_flip_all_ones(qc: QuantumCircuit, data, anc=None):
    """
    对 data 上的态 |11...1> 加 -1 相位。
    - anc=None：在数据位上直接做多控 Z（H(target)-MCX-H(target)）；n=1 时退化为 Z。
    - anc!=None：假设 anc 已置于 |->，执行 MCX(data -> anc) 进行相位回传。
    """
    n = len(data)
    if anc is not None:
        qc.mcx(data, anc)
    else:
        if n == 1:
            qc.z(data[0])
        else:
            tgt = data[-1]
            ctrls = data[:-1]
            qc.h(tgt); qc.mcx(ctrls, tgt); qc.h(tgt)

def _oracle_mark_solutions(qc: QuantumCircuit, data, solutions_bits, anc=None):
    """
    Oracle：对 solutions_bits（列表，每个是小端 0/1 列表）中的所有解翻相位。
    通过把解的 0 位先 X → 映射到全 1，然后调用 _phase_flip_all_ones，再还原。
    """
    for bits in solutions_bits:
        zeros = [i for i, b in enumerate(bits) if b == 0]
        for i in zeros: qc.x(data[i])
        _phase_flip_all_ones(qc, data, anc=anc)
        for i in zeros: qc.x(data[i])

def _diffuser_about_plus(qc: QuantumCircuit, data):
    """扩散算子 D：关于 |++...+> 的反射：H^n X^n  · MCZ ·  X^n H^n。"""
    n = len(data)
    qc.h(data); qc.x(data)
    _phase_flip_all_ones(qc, data, anc=None)
    qc.x(data); qc.h(data)


# ---------- 公开 API 1：只构建电路 ----------

def build_grover_circuit(
    n_data: int,
    solutions: list[str],
    rounds: int | None = None,
    use_ancilla: bool = False,
    measure: str | list[int] = "data",
):
    """
    构造 Grover“标记-扩散”电路（多解可选）。

    参数：
    - n_data: 数据位数量（>=1）
    - solutions: 解的比特串列表（高位在左，如 '101'）。若传空，将只构造扩散器反射（一般无意义）。
    - rounds: 迭代次数；None 时按 floor(pi/4 * sqrt(N/M)) 自动设置，其中 N=2^n, M=len(solutions)。
    - use_ancilla: 是否使用一个辅助位做相位回传（anc 放在最后一位，并初始化到 |->）。
    - measure: "data"（只测数据位，默认），"all"（全测），或量子位索引列表。

    返回：
    - QuantumCircuit：已根据 measure 选择是否附带测量。
    """
    if n_data < 1:
        raise ValueError("n_data 必须 ≥ 1")
    N = 2**n_data
    M = max(1, len(solutions))
    if rounds is None:
        rounds = int(np.floor(np.pi/4 * np.sqrt(N / M)))

    # 小端化目标解
    sols_bits = [_bitstr_to_little_endian_bits(s, n_data) for s in solutions]

    # 量子位布置：data = [0..n-1]；anc = n（若启用）
    if use_ancilla:
        qc = QuantumCircuit(n_data + 1)
        data = list(range(n_data))
        anc = n_data
        # 初始化：数据位 → 均匀态 |++...+>；辅助位 → |->（用于相位回传）
        qc.h(data)
        qc.x(anc); qc.h(anc)
    else:
        qc = QuantumCircuit(n_data)
        data = list(range(n_data))
        anc = None
        qc.h(data)

    # r 次 Grover 迭代：Oracle → Diffuser
    for _ in range(rounds):
        if sols_bits:
            _oracle_mark_solutions(qc, data, sols_bits, anc=anc)
        _diffuser_about_plus(qc, data)

    # 测量布置
    if measure == "data":
        qcm = QuantumCircuit(qc.num_qubits, len(data))
        qcm.compose(qc, inplace=True)
        qcm.measure(data, range(len(data)))
        return qcm
    elif measure == "all":
        qcm = QuantumCircuit(qc.num_qubits, qc.num_qubits)
        qcm.compose(qc, inplace=True)
        qcm.measure(range(qc.num_qubits), range(qc.num_qubits))
        return qcm
    elif isinstance(measure, (list, tuple)):
        cols = list(measure)
        qcm = QuantumCircuit(qc.num_qubits, len(cols))
        qcm.compose(qc, inplace=True)
        qcm.measure(cols, range(len(cols)))
        return qcm
    else:
        return qc  # 不加测量


# ---------- 公开 API 2：构建+运行+出图 ----------

def run_grover(
    n_data: int,
    solutions: list[str],
    rounds: int | None = None,
    use_ancilla: bool = False,
    shots: int = 20_000,
    plot_prob: bool = True,
    seed: int | None = 1,
):
    """
    一把梭：构建 Grover 电路，仿真运行，并画出计数/概率直方图。
    返回 (qc, counts, probs)

    备注：直方图为小端序标签；例如 '101' 表示 q2 q1 q0 = 1 0 1。
    """
    qc = build_grover_circuit(
        n_data=n_data,
        solutions=solutions,
        rounds=rounds,
        use_ancilla=use_ancilla,
        measure="data",
    )
    backend = Aer.get_backend("aer_simulator")
    if seed is not None:
        backend.set_options(seed_simulator=seed)
    res = backend.run(transpile(qc, backend), shots=shots).result()
    counts = res.get_counts()
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    if plot_prob:
        plot_distribution(probs, title=f"Grover Probabilities (n={n_data}, sols={solutions})")
    else:
        plot_histogram(counts, title=f"Grover Counts (n={n_data}, sols={solutions})")
    plt.show()

    return qc, counts, probs


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 示例 1：3 比特、唯一解 '101'，自动选择最优迭代次数
    qc, counts, probs = run_grover(n_data=3, solutions=["101"], use_ancilla=False)

    # 示例 2：2 比特、两个解 '01' 与 '10'，手动做 1 轮，使用辅助位
    qc2 = build_grover_circuit(n_data=2, solutions=["01", "10"], rounds=1,
                               use_ancilla=True, measure="data")
    backend = Aer.get_backend("aer_simulator")
    res2 = backend.run(transpile(qc2, backend), shots=20000).result()
    plot_histogram(res2.get_counts(), title="Two solutions, 1 iteration")
    plt.show()
