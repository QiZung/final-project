import numpy as np
import matplotlib.pyplot as plt

# 定义不同的步长
h_values = [0.05, 0.1,0.2,0.25,0.5]

# 初始化误差列表
errors = []

for h in h_values:
    # 定义网格参数
    nx = int(1 / h)  # x轴方向上的网格数量
    ny = int(1 / h)  # y轴方向上的网格数量
    dx = h  # x轴方向上的网格间距
    dy = h  # y轴方向上的网格间距

    # 定义边界条件
    u_boundary = np.zeros((nx + 2, ny + 2))
    u_boundary[0, :] = 0  # 下边界
    u_boundary[-1, :] = 0  # 上边界
    u_boundary[:, 0] = 0  # 左边界
    u_boundary[:, -1] = 0  # 右边界

    # 初始化解向量
    u = np.zeros((nx + 2, ny + 2))

    # 定义右侧项
    f = np.zeros((nx + 2, ny + 2))
    x = np.linspace(0, 1, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    X, Y = np.meshgrid(x, y)
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # 定义精确解
    exact_u = np.zeros((nx + 2, ny + 2))
    for i in range(nx + 2):
        for j in range(ny + 2):
            exact_u[i, j] = (-1 / (2 * np.pi**2)) * np.sin(np.pi * i * dx) * np.sin(np.pi * j * dy)

    # 迭代求解
    max_iter = 1000  # 最大迭代次数
    tolerance = 1e-4  # 收敛容差

    for iteration in range(max_iter):
        u_old = u.copy()

        # 使用有限差分方法更新解
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                u[i, j] = (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1] - dx**2 * f[i, j]) / 4

        # 检查收敛性
        residual = np.max(np.abs(u - u_old))
        if residual < tolerance:
            break

    # 计算误差
    error = np.abs(u[1:-1, 1:-1] - exact_u[1:-1, 1:-1])

    # 计算最大误差
    max_error = np.max(error)
    errors.append(max_error)

# 对步长和误差取对数
log_h_values = np.log10(h_values)
log_errors = np.log10(errors)

# 拟合直线
coeffs = np.polyfit(log_h_values, log_errors, 1)
poly = np.poly1d(coeffs)

# 计算斜率
slope = coeffs[0]

# 绘制对数图形和拟合直线
plt.plot(log_h_values, log_errors, marker='o', label='Data')
plt.plot(log_h_values, poly(log_h_values), label='Fit')
plt.xlabel('log(Step Size)')
plt.ylabel('log(Maximum Error)')
plt.title('Log(Error) vs. Log(Step Size)')
plt.legend()

# 显示斜率
plt.text(log_h_values[0], log_errors[0], f'Slope: {slope:.2f}', ha='left', va='bottom')

plt.grid(True)
plt.show()