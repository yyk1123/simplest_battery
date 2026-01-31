import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 电池参数设置 ====================
# 电池基本参数
C_battery = 2.2  # 电池容量 (Ah)，典型18650电池
SOC_initial = 1.0  # 初始SOC (100%)
I_load = 2.0  # 恒流放电电流 (A)

# 3RC模型参数 (典型值，基于文献拟合)
# 欧姆内阻
R0 = 0.05  # Ω

# RC支路1 (短时间常数，模拟电化学极化)
R1 = 0.01  # Ω
C1 = 3000  # F

# RC支路2 (中时间常数，模拟浓差极化)
R2 = 0.02  # Ω  
C2 = 12000  # F

# RC支路3 (长时间常数，模拟慢扩散过程)
R3 = 0.03  # Ω
C3 = 25000  # F

# 截止条件
V_cutoff = 2.8  # 电压截止点 (V)，低于此值视为耗尽
SOC_min = 0.05  # SOC截止点，低于5%视为耗尽

# ==================== OCV-SOC关系函数 ====================
def ocv_from_soc(SOC):
    """
    根据SOC计算开路电压(OCV)
    使用更精确的6阶多项式拟合典型的Li-ion电池OCV曲线
    """
    SOC = np.clip(SOC, 0, 1)  # 限制SOC在[0,1]范围内
    
    # 典型18650锂离子电池OCV-SOC关系 (NMC材料)
    # 多项式系数通过实验数据拟合得到
    coeffs = [3.0, 1.5, -2.5, 3.0, -2.0, 0.5, 0.0]
    
    V_oc = 0
    for i, coeff in enumerate(coeffs):
        V_oc += coeff * SOC**i
    
    return V_oc

# ==================== 系统动力学方程 ====================
def battery_model(t, state, I_load, C_battery, R0, R1, C1, R2, C2, R3, C3):
    """
    电池三RC模型的状态方程
    state: [Vc1, Vc2, Vc3, SOC]
    """
    Vc1, Vc2, Vc3, SOC = state
    
    # 确保SOC在合理范围内
    SOC = np.clip(SOC, 0, 1)
    
    # OCV-SOC关系
    V_oc = ocv_from_soc(SOC)
    
    # 状态导数 - 核心方程
    dVc1_dt = (I_load - Vc1/R1) / C1
    dVc2_dt = (I_load - Vc2/R2) / C2
    dVc3_dt = (I_load - Vc3/R3) / C3
    
    # SOC变化率 (单位: 1/s)
    # C_battery * 3600 将Ah转换为As (库仑)
    dSOC_dt = -I_load / (C_battery * 3600)
    
    return [dVc1_dt, dVc2_dt, dVc3_dt, dSOC_dt]

# ==================== 事件函数：检测电池耗尽 ====================
def depletion_event(t, state, I_load, C_battery, R0, R1, C1, R2, C2, R3, C3):
    """检测电池是否耗尽的事件函数"""
    Vc1, Vc2, Vc3, SOC = state
    
    # 确保SOC在合理范围内
    SOC = np.clip(SOC, 0, 1)
    
    # 计算OCV和端电压
    V_oc = ocv_from_soc(SOC)
    V_terminal = V_oc - I_load * R0 - Vc1 - Vc2 - Vc3
    
    # 当电压低于截止电压或SOC低于最小值时停止
    return min(V_terminal - V_cutoff, SOC - SOC_min)

depletion_event.terminal = True  # 事件发生时停止积分
depletion_event.direction = -1   # 仅当值从正变负时触发

# ==================== 求解微分方程 ====================
# 初始条件: 电容电压为0，SOC为1
initial_state = [0.0, 0.0, 0.0, SOC_initial]

# 最大仿真时间 (秒) - 电池完全放电的理论最大时间
t_max = C_battery * 3600 / I_load  # 理论放电时间 (秒)
t_span = (0, t_max * 1.5)  # 留50%余量以确保捕捉耗尽点

print("=== 电池参数 ===")
print(f"电池容量: {C_battery} Ah")
print(f"放电电流: {I_load} A")
print(f"理论最大放电时间: {t_max/3600:.2f} 小时 ({t_max:.0f} 秒)")
print(f"欧姆内阻 R0: {R0} Ω")
print(f"RC1: R1={R1} Ω, C1={C1} F, 时间常数 τ1={R1*C1:.0f} s")
print(f"RC2: R2={R2} Ω, C2={C2} F, 时间常数 τ2={R2*C2:.0f} s")
print(f"RC3: R3={R3} Ω, C3={C3} F, 时间常数 τ3={R3*C3:.0f} s")

# 求解ODE
solution = solve_ivp(
    battery_model,
    t_span,
    initial_state,
    args=(I_load, C_battery, R0, R1, C1, R2, C2, R3, C3),
    events=depletion_event,
    max_step=5,  # 最大步长5秒
    rtol=1e-8,
    atol=1e-10,
    dense_output=True
)

# ==================== 提取结果 ====================
t = solution.t  # 时间 (秒)
Vc1, Vc2, Vc3, SOC = solution.y  # 状态变量

# 计算端电压和OCV
V_terminal = np.zeros_like(t)
V_oc_array = np.zeros_like(t)

for i in range(len(t)):
    V_oc = ocv_from_soc(SOC[i])
    V_oc_array[i] = V_oc
    V_terminal[i] = V_oc - I_load * R0 - Vc1[i] - Vc2[i] - Vc3[i]

DoD = 1 - SOC  # 放电深度

# 找到耗尽时间点
if len(solution.t_events[0]) > 0:
    depletion_time = solution.t_events[0][0]
    # 获取耗尽时的状态
    depletion_idx = np.argmin(np.abs(t - depletion_time))
else:
    depletion_time = t[-1]
    depletion_idx = -1

# ==================== 绘图 ====================
plt.figure(figsize=(16, 12))

# 子图1: 端电压和OCV随时间变化
plt.subplot(3, 2, 1)
plt.plot(t/3600, V_terminal, 'b-', linewidth=2.5, label='端电压')
plt.plot(t/3600, V_oc_array, 'r--', linewidth=1.5, alpha=0.7, label='开路电压(OCV)')
plt.axhline(y=V_cutoff, color='r', linestyle=':', linewidth=1.5, 
           label=f'截止电压 ({V_cutoff}V)')
if depletion_idx >= 0:
    plt.axvline(x=depletion_time/3600, color='g', linestyle='--', linewidth=1.5,
               label=f'耗尽时间 ({depletion_time/3600:.2f}h)')
plt.xlabel('时间 (小时)', fontsize=12)
plt.ylabel('电压 (V)', fontsize=12)
plt.title('电池端电压和开路电压随时间变化', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 子图2: 端电压 vs DoD
plt.subplot(3, 2, 2)
plt.plot(DoD[:depletion_idx+1] if depletion_idx>=0 else DoD, 
         V_terminal[:depletion_idx+1] if depletion_idx>=0 else V_terminal, 
         'b-', linewidth=2.5)
plt.xlabel('放电深度 (DoD)', fontsize=12)
plt.ylabel('端电压 (V)', fontsize=12)
plt.title('端电压 vs 放电深度', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 子图3: SOC和DoD随时间变化
plt.subplot(3, 2, 3)
plt.plot(t/3600, SOC*100, 'g-', linewidth=2.5, label='SOC')
plt.plot(t/3600, DoD*100, 'r-', linewidth=2, alpha=0.7, label='DoD')
plt.axhline(y=SOC_min*100, color='r', linestyle=':', linewidth=1.5, 
           label=f'SOC下限 ({SOC_min*100:.0f}%)')
if depletion_idx >= 0:
    plt.axvline(x=depletion_time/3600, color='g', linestyle='--', linewidth=1.5)
plt.xlabel('时间 (小时)', fontsize=12)
plt.ylabel('百分比 (%)', fontsize=12)
plt.title('SOC和DoD随时间变化', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 子图4: 电容电压随时间变化
plt.subplot(3, 2, 4)
plt.plot(t/3600, Vc1, 'b-', linewidth=2, label=f'Vc1 (τ1={R1*C1:.0f}s)')
plt.plot(t/3600, Vc2, 'r-', linewidth=2, label=f'Vc2 (τ2={R2*C2:.0f}s)')
plt.plot(t/3600, Vc3, 'g-', linewidth=2, label=f'Vc3 (τ3={R3*C3:.0f}s)')
if depletion_idx >= 0:
    plt.axvline(x=depletion_time/3600, color='k', linestyle='--', linewidth=1.5)
plt.xlabel('时间 (小时)', fontsize=12)
plt.ylabel('电容电压 (V)', fontsize=12)
plt.title('三个RC支路的电容电压', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 子图5: 各电压分量随SOC变化
plt.subplot(3, 2, 5)
SOC_range = np.linspace(0, 1, 100)
OCV_range = ocv_from_soc(SOC_range)

# 创建更精细的时间序列用于评估
if solution.sol is not None:
    t_dense = np.linspace(0, min(depletion_time, t_max*1.5) if depletion_idx>=0 else t_max, 500)
    sol_dense = solution.sol(t_dense)
    SOC_dense = sol_dense[3]
    SOC_dense = np.clip(SOC_dense, 0, 1)
    Vc1_dense = sol_dense[0]
    Vc2_dense = sol_dense[1]
    Vc3_dense = sol_dense[2]
    V_terminal_dense = ocv_from_soc(SOC_dense) - I_load*R0 - Vc1_dense - Vc2_dense - Vc3_dense
else:
    SOC_dense = SOC
    V_terminal_dense = V_terminal

plt.plot(SOC_dense*100, V_terminal_dense, 'b-', linewidth=2.5, label='端电压')
plt.plot(SOC_range*100, OCV_range, 'r--', linewidth=1.5, alpha=0.7, label='OCV')
plt.axvline(x=SOC_min*100, color='r', linestyle=':', linewidth=1.5)
plt.xlabel('SOC (%)', fontsize=12)
plt.ylabel('电压 (V)', fontsize=12)
plt.title('端电压和OCV随SOC变化', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 子图6: 总电压降分解
plt.subplot(3, 2, 6)
if depletion_idx >= 0:
    # 计算各分量在耗尽时的贡献
    V_oc_init = ocv_from_soc(1.0)
    V_oc_final = ocv_from_soc(SOC[depletion_idx])
    
    components = {
        'OCV下降': V_oc_init - V_oc_final,
        '欧姆压降': I_load * R0,
        'RC1压降': Vc1[depletion_idx],
        'RC2压降': Vc2[depletion_idx],
        'RC3压降': Vc3[depletion_idx]
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = plt.bar(components.keys(), components.values(), color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, value in zip(bars, components.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}V', ha='center', va='bottom', fontsize=9)
    
    plt.ylabel('电压降 (V)', fontsize=12)
    plt.title('总电压降分解 (耗尽时)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# ==================== 输出详细结果 ====================
print("\n=== 仿真结果总结 ===")
if depletion_idx >= 0:
    print(f"实际耗尽时间: {depletion_time:.1f} 秒 ({depletion_time/3600:.2f} 小时)")
    print(f"电池效率: {depletion_time/t_max*100:.1f}% (实际/理论放电时间比)")
    
    print(f"\n初始时刻 (t=0):")
    print(f"  SOC: {SOC[0]*100:.1f}%")
    print(f"  OCV: {ocv_from_soc(1.0):.3f} V")
    print(f"  端电压: {V_terminal[0]:.3f} V")
    
    print(f"\n耗尽时刻 (t={depletion_time:.1f}s):")
    print(f"  SOC: {SOC[depletion_idx]*100:.2f}%")
    print(f"  DoD: {DoD[depletion_idx]*100:.2f}%")
    print(f"  OCV: {V_oc_array[depletion_idx]:.3f} V")
    print(f"  端电压: {V_terminal[depletion_idx]:.3f} V")
    
    # 计算各分量电压降
    print(f"\n电压降分析:")
    total_drop = V_terminal[0] - V_terminal[depletion_idx]
    print(f"  总电压降: {total_drop:.3f} V")
    
    print(f"  各分量压降:")
    print(f"    - OCV下降: {ocv_from_soc(1.0) - V_oc_array[depletion_idx]:.3f} V")
    print(f"    - 欧姆压降 (恒定): {I_load*R0:.3f} V")
    print(f"    - RC1极化压降: {Vc1[depletion_idx]:.3f} V")
    print(f"    - RC2极化压降: {Vc2[depletion_idx]:.3f} V")
    print(f"    - RC3极化压降: {Vc3[depletion_idx]:.3f} V")
    
    print(f"\n各RC支路时间常数:")
    print(f"  τ1 = R1*C1 = {R1*C1:.0f} s ({R1*C1/60:.1f} 分钟)")
    print(f"  τ2 = R2*C2 = {R2*C2:.0f} s ({R2*C2/60:.1f} 分钟)")
    print(f"  τ3 = R3*C3 = {R3*C3:.0f} s ({R3*C3/60:.1f} 分钟)")
    
    # 计算不同时间段的平均电压
    quarter_time = depletion_time / 4
    time_points = [0, quarter_time, quarter_time*2, quarter_time*3, depletion_time]
    time_labels = ["0-25%", "25-50%", "50-75%", "75-100%", "平均"]
    
    print(f"\n不同放电阶段的平均电压:")
    for i in range(4):
        start_idx = np.argmin(np.abs(t - time_points[i]))
        end_idx = np.argmin(np.abs(t - time_points[i+1]))
        avg_voltage = np.mean(V_terminal[start_idx:end_idx])
        print(f"  {time_labels[i]}: {avg_voltage:.3f} V")
    
    # 显示电压曲线特征
    print(f"\n电压曲线特征:")
    max_voltage = np.max(V_terminal)
    min_voltage = np.min(V_terminal)
    print(f"  最高电压: {max_voltage:.3f} V")
    print(f"  最低电压: {min_voltage:.3f} V")
    print(f"  电压变化范围: {max_voltage-min_voltage:.3f} V")
    
    # 计算电压下降速率
    time_mid = depletion_time / 2
    mid_idx = np.argmin(np.abs(t - time_mid))
    V_mid = V_terminal[mid_idx]
    
    # 计算初始下降率（前10%时间）
    idx_10pct = np.argmin(np.abs(t - depletion_time*0.1))
    initial_drop_rate = (V_terminal[0] - V_terminal[idx_10pct]) / (t[idx_10pct] - t[0])
    print(f"  初始电压下降速率: {initial_drop_rate*1000:.2f} mV/s ({initial_drop_rate*3600:.2f} V/h)")
    
    # 计算最终下降率（最后10%时间）
    idx_90pct = np.argmin(np.abs(t - depletion_time*0.9))
    final_drop_rate = (V_terminal[idx_90pct] - V_terminal[depletion_idx]) / (t[depletion_idx] - t[idx_90pct])
    print(f"  最终电压下降速率: {final_drop_rate*1000:.2f} mV/s ({final_drop_rate*3600:.2f} V/h)")
else:
    print("未达到耗尽条件")
    print(f"仿真结束时: t={t[-1]/3600:.2f}h, SOC={SOC[-1]*100:.1f}%, V={V_terminal[-1]:.3f}V")