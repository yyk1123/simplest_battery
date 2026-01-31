import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==================== 绘图设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 7))

# 关闭坐标轴
ax.set_xlim(-1, 15)
ax.set_ylim(-3, 3)
ax.axis('off')
ax.set_aspect('equal')
ax.set_title('锂电池 3RC 等效电路模型', fontsize=18, fontweight='bold', pad=20)

# ==================== 辅助函数 ====================
def draw_resistor_h(ax, x_start, y, width=1.0, label='', label_pos='top'):
    """绘制水平电阻（锯齿形）"""
    n_teeth = 6
    x_points = np.linspace(x_start, x_start + width, n_teeth * 2 + 1)
    y_points = [y]
    for i in range(1, len(x_points) - 1):
        if i % 2 == 1:
            y_points.append(y + 0.15)
        else:
            y_points.append(y - 0.15)
    y_points.append(y)
    ax.plot(x_points, y_points, 'k-', lw=1.5)
    
    if label:
        label_y = y + 0.35 if label_pos == 'top' else y - 0.35
        va = 'bottom' if label_pos == 'top' else 'top'
        ax.text(x_start + width/2, label_y, label, ha='center', va=va, fontsize=12, fontweight='bold')
    return x_start + width

def draw_resistor_v(ax, x, y_start, height=1.0, label='', label_pos='right'):
    """绘制垂直电阻（锯齿形）"""
    n_teeth = 6
    y_points = np.linspace(y_start, y_start - height, n_teeth * 2 + 1)
    x_points = [x]
    for i in range(1, len(y_points) - 1):
        if i % 2 == 1:
            x_points.append(x + 0.15)
        else:
            x_points.append(x - 0.15)
    x_points.append(x)
    ax.plot(x_points, y_points, 'k-', lw=1.5)
    
    if label:
        label_x = x + 0.35 if label_pos == 'right' else x - 0.35
        ha = 'left' if label_pos == 'right' else 'right'
        ax.text(label_x, y_start - height/2, label, ha=ha, va='center', fontsize=12, fontweight='bold')
    return y_start - height

def draw_capacitor_v(ax, x, y_start, gap=0.15, plate_width=0.4, label='', label_pos='right'):
    """绘制垂直电容"""
    y_mid = y_start - 0.5
    # 上极板
    ax.plot([x - plate_width/2, x + plate_width/2], [y_mid + gap/2, y_mid + gap/2], 'k-', lw=2.5)
    # 下极板
    ax.plot([x - plate_width/2, x + plate_width/2], [y_mid - gap/2, y_mid - gap/2], 'k-', lw=2.5)
    # 连线
    ax.plot([x, x], [y_start, y_mid + gap/2], 'k-', lw=1.5)
    ax.plot([x, x], [y_mid - gap/2, y_start - 1.0], 'k-', lw=1.5)
    
    if label:
        label_x = x + 0.35 if label_pos == 'right' else x - 0.35
        ha = 'left' if label_pos == 'right' else 'right'
        ax.text(label_x, y_mid, label, ha=ha, va='center', fontsize=12, fontweight='bold', color='blue')
    return y_start - 1.0

def draw_voltage_source(ax, x, y_center, radius=0.4, label=''):
    """绘制电压源（圆圈带+/-）"""
    circle = patches.Circle((x, y_center), radius, fill=False, color='black', lw=2)
    ax.add_patch(circle)
    # 画 +/- 号
    ax.plot([x, x], [y_center + 0.15, y_center + 0.25], 'k-', lw=2)  # + 竖线
    ax.plot([x - 0.05, x + 0.05], [y_center + 0.2, y_center + 0.2], 'k-', lw=2)  # + 横线
    ax.plot([x - 0.05, x + 0.05], [y_center - 0.2, y_center - 0.2], 'k-', lw=2)  # - 横线
    
    if label:
        ax.text(x - 0.6, y_center, label, ha='right', va='center', fontsize=11, fontweight='bold')

def draw_rc_parallel(ax, x_center, y_top, y_bottom, r_label='R', c_label='C'):
    """绘制并联RC支路"""
    branch_width = 0.8  # 两条支路的间距
    height = y_top - y_bottom
    
    # 上方分叉点
    ax.plot(x_center, y_top, 'ko', markersize=5, zorder=5)
    # 下方汇合点
    ax.plot(x_center, y_bottom, 'ko', markersize=5, zorder=5)
    
    # 左支路 - 电阻
    x_left = x_center - branch_width/2
    ax.plot([x_center, x_left], [y_top, y_top], 'k-', lw=1.5)  # 水平连接到左边
    ax.plot([x_left, x_left], [y_top, y_top - 0.3], 'k-', lw=1.5)  # 上垂直线
    draw_resistor_v(ax, x_left, y_top - 0.3, height=height - 0.6, label=r_label, label_pos='left')
    ax.plot([x_left, x_left], [y_bottom + 0.3, y_bottom], 'k-', lw=1.5)  # 下垂直线
    ax.plot([x_left, x_center], [y_bottom, y_bottom], 'k-', lw=1.5)  # 水平连接回来
    
    # 右支路 - 电容
    x_right = x_center + branch_width/2
    ax.plot([x_center, x_right], [y_top, y_top], 'k-', lw=1.5)  # 水平连接到右边
    ax.plot([x_right, x_right], [y_top, y_top - 0.3], 'k-', lw=1.5)  # 上垂直线
    
    # 电容
    y_cap_top = y_top - 0.3
    y_cap_mid = (y_top + y_bottom) / 2
    plate_width = 0.35
    gap = 0.12
    ax.plot([x_right, x_right], [y_cap_top, y_cap_mid + gap/2], 'k-', lw=1.5)
    ax.plot([x_right - plate_width/2, x_right + plate_width/2], [y_cap_mid + gap/2, y_cap_mid + gap/2], 'k-', lw=2.5)
    ax.plot([x_right - plate_width/2, x_right + plate_width/2], [y_cap_mid - gap/2, y_cap_mid - gap/2], 'k-', lw=2.5)
    ax.plot([x_right, x_right], [y_cap_mid - gap/2, y_bottom + 0.3], 'k-', lw=1.5)
    ax.plot([x_right, x_right], [y_bottom + 0.3, y_bottom], 'k-', lw=1.5)
    ax.plot([x_right, x_center], [y_bottom, y_bottom], 'k-', lw=1.5)
    
    # 电容标签
    ax.text(x_right + 0.3, y_cap_mid, c_label, ha='left', va='center', fontsize=12, fontweight='bold', color='blue')

# ==================== 开始绘制电路 ====================
# 定义关键坐标
y_top = 1.5       # 上方导线
y_bottom = -1.5   # 下方导线
x_start = 0.5     # 起始位置

# 1. 绘制电压源 Vocv
x_vocv = x_start
ax.plot([x_vocv, x_vocv], [y_top, 0.4], 'k-', lw=1.5)  # 上方连线
draw_voltage_source(ax, x_vocv, 0, radius=0.4, label=r'$V_{ocv}$(SOC)')
ax.plot([x_vocv, x_vocv], [-0.4, y_bottom], 'k-', lw=1.5)  # 下方连线

# 2. 绘制上方主导线和R0
x_r0_start = x_vocv + 0.8
ax.plot([x_vocv, x_r0_start], [y_top, y_top], 'k-', lw=1.5)  # 连接到R0

# R0 电阻
x_r0_end = draw_resistor_h(ax, x_r0_start, y_top, width=1.2, label=r'$R_0$')

# 3. 连接到第一个RC并联支路
x_rc1 = x_r0_end + 1.5
ax.plot([x_r0_end, x_rc1], [y_top, y_top], 'k-', lw=1.5)
draw_rc_parallel(ax, x_rc1, y_top, y_bottom, r_label=r'$R_1$', c_label=r'$C_1$')

# 4. 第二个RC并联支路
x_rc2 = x_rc1 + 2.0
ax.plot([x_rc1, x_rc2], [y_top, y_top], 'k-', lw=1.5)
ax.plot([x_rc1, x_rc2], [y_bottom, y_bottom], 'k-', lw=1.5)
draw_rc_parallel(ax, x_rc2, y_top, y_bottom, r_label=r'$R_2$', c_label=r'$C_2$')

# 5. 第三个RC并联支路
x_rc3 = x_rc2 + 2.0
ax.plot([x_rc2, x_rc3], [y_top, y_top], 'k-', lw=1.5)
ax.plot([x_rc2, x_rc3], [y_bottom, y_bottom], 'k-', lw=1.5)
draw_rc_parallel(ax, x_rc3, y_top, y_bottom, r_label=r'$R_3$', c_label=r'$C_3$')

# 6. 连接到输出端子
x_terminal = x_rc3 + 1.5
ax.plot([x_rc3, x_terminal], [y_top, y_top], 'k-', lw=1.5)
ax.plot([x_rc3, x_terminal], [y_bottom, y_bottom], 'k-', lw=1.5)

# 绘制端子符号
ax.plot([x_terminal, x_terminal + 0.3], [y_top, y_top], 'k-', lw=2)
ax.plot([x_terminal, x_terminal + 0.3], [y_bottom, y_bottom], 'k-', lw=2)
ax.plot(x_terminal + 0.3, y_top, 'ko', markersize=8)
ax.plot(x_terminal + 0.3, y_bottom, 'ko', markersize=8)

# 端子标注
ax.text(x_terminal + 0.5, y_top, r'+', fontsize=16, va='center', ha='left', fontweight='bold')
ax.text(x_terminal + 0.5, y_bottom, r'-', fontsize=16, va='center', ha='left', fontweight='bold')
ax.text(x_terminal + 1.0, (y_top + y_bottom)/2, r'$V_{terminal}$', fontsize=13, va='center', ha='left', fontweight='bold')

# 7. 下方导线连接回电压源
ax.plot([x_vocv, x_rc1], [y_bottom, y_bottom], 'k-', lw=1.5)

# 8. 添加电流方向箭头
arrow_y = y_top + 0.4
ax.annotate('', xy=(x_r0_start + 0.6, arrow_y), xytext=(x_r0_start, arrow_y),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(x_r0_start + 0.3, arrow_y + 0.25, r'$I$', fontsize=12, ha='center', color='red', fontweight='bold')

# 显示图形
plt.tight_layout()
plt.show()