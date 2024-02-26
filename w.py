import numpy as np
import matplotlib.pyplot as plt


def generate_plots():
    xa, xb = 0.0, 1.0
    ya, yb = 0.0, 1.0

    h = 0.05
    # w_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    w_values = [1.0, 1.2, 1.4,1.6]
    nodes = round((xb - xa) / h)

    for idx, w in enumerate(w_values):
        u = np.zeros((nodes + 1, nodes + 1))

        # Iterative relaxation method
        for _ in range(100):
            for i in range(1, nodes):
                for j in range(1, nodes):
                    u[i, j] = w / 4 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - (h**2) * np.sin(np.pi * i * h) * np.sin(np.pi * j * h)) + (1 - w) * u[i, j] 

        # Plotting
        x = np.linspace(0, 1, nodes + 1) 
        y = np.linspace(0, 1, nodes + 1)
        xx, yy = np.meshgrid(x, y)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, u, cmap=plt.get_cmap('rainbow'))
        ax.set_title(f'Approximated Solution (w = {w})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.text(0.1, 0.1, 0.8, f'h = {h}', color='black', fontsize=10)

        plt.show()


generate_plots()