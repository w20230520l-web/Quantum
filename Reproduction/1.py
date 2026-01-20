def visualize_comprehensive_circuit():
    print("\n" + "=" * 50)
    print("🗺️ [Phase 2] 正在生成全景细节线路图...")
    print("=" * 50)

    # --- A. 准备数学参数 ---
    # 我们的归一化矩阵 A = [[0.75, 0.25], [0.25, 0.75]]
    # 核心元素是 0.75
    matrix_element = 0.75

    # 计算量子门的旋转角度 theta
    # Block Encoding 原理: A_00 = cos(theta/2)
    # 所以 theta = 2 * arccos(A_00)
    theta = 2 * np.arccos(matrix_element)

    print(f"1. 矩阵主元素 A[0,0]: {matrix_element}")
    print(f"2. 映射为量子参数 theta: {theta:.4f} rad")
    print("   (你将在图中紫色的方块上看到这个数字 1.445)")

     # --- B. 定义量子寄存器 ---
    qr_sys = QuantumRegister(1, name="System (x)")
    qr_anc = QuantumRegister(1, name="Ancilla (Ctrl)")
    qc = QuantumCircuit(qr_sys, qr_anc)

    # --- C. 定义 LCU (加法器) 的构建模块 ---
    def append_lcu_step(circuit, sys, anc, angle, label_prefix):
        # 1. 分身 (Hadamard) - 开启并行宇宙
        circuit.h(anc)

        # 2. 路径选择 (Ancilla=1 时触发)
        circuit.x(anc)

        # 3. 核心：受控旋转 (CUGate)
        # 这就是矩阵 A 的物理身躯！
        # 它的参数 angle 就是我们算的 theta
        cu_gate = CUGate(angle, 0, 0, 0, label=f"{label_prefix}Grad_Op")
        circuit.append(cu_gate, [anc, sys])

        circuit.x(anc)

        # 4. 干涉 (Hadamard) - 合并宇宙
        circuit.h(anc)

    # --- D. 组装流水线 ---

    # [第 0 阶段]: 初始化
    qc.reset(qr_sys)
    qc.reset(qr_anc)
    qc.barrier(label="Init x0")  # <--- 视觉路标

    # [第 1 阶段]: 第一次迭代
    append_lcu_step(qc, qr_sys[0], qr_anc[0], theta, label_prefix="Iter1_")
    qc.barrier(label="End of Iter 1")  # <--- 视觉路标

    # [第 2 阶段]: 第二次迭代 (重复结构)
    append_lcu_step(qc, qr_sys[0], qr_anc[0], theta, label_prefix="Iter2_")
    qc.barrier(label="End of Iter 2")  # <--- 视觉路标

    # [收尾]: 测量
    qc.measure_all()

    # --- E. 绘图 (关键设置) ---
    print("3. 正在渲染超长全景图...")

    # fold=-1: 强制不折行，画成一条长龙，看清所有流程
    # scale=0.8: 稍微缩小，防止图太大
    # style='iqp': 最美观的现代风格
    qc.draw(output='mpl', style='iqp', fold=-1, scale=0.8)

    plt.title(f"Quantum Linear Solver Circuit (Flow + Details)\nTheta={theta:.3f} encodes A=0.75", fontsize=14,
                y=1.1)
    plt.show()
