import numpy as np
import matplotlib.pyplot as plt


def generate_plots():
    xa, xb = 0.0, 1.0
    ya, yb = 0.0, 1.0

    h = 0.05
    w_values = [0.6,0.8,1.0, 1.2, 1.4, 1.6]
   
    nodes = round((xb - xa) / h)

    errors = []  # 用于存储每个 w 值对应的误差

    for idx, w in enumerate(w_values):
        u = np.zeros((nodes + 1, nodes + 1))

        # 迭代松弛方法
        for _ in range(100):
            for i in range(1, nodes):
                for j in range(1, nodes):
                    u[i, j] = w / 4 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - (h**2) * np.sin(np.pi * i * h) * np.sin(np.pi * j * h)) + (1 - w) * u[i, j]

        # 计算误差
        exact_u = np.zeros((nodes + 1, nodes + 1))
        for i in range(nodes + 1):
            for j in range(nodes + 1):
                exact_u[i, j] = (-1 / (2 * np.pi**2)) * np.sin(np.pi * i * h) * np.sin(np.pi * j * h)
        error = np.max(np.abs(u - exact_u))
        errors.append(error)

    # 绘制误差曲线
    fig, ax = plt.subplots()
    ax.plot(w_values, errors, marker='o')
    ax.set_title('error vs w')
    ax.set_xlabel('w')
    ax.set_ylabel('biggest error')
    ax.grid(True)

    plt.show()


generate_plots()