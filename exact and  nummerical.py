import numpy as np
import matplotlib.pyplot as plt


xa, xb = 0.0, 1.0
ya, yb = 0.0, 1.0

h = 0.05
w = 1.0
nodes = round((xb - xa) / h)

u = np.zeros((nodes + 1, nodes + 1))

# Iterative relaxation method
for _ in range(100):
    for i in range(1, nodes):
        for j in range(1, nodes):
            u[i, j] = w / 4 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - (h**2) * np.sin(np.pi * i * h) * np.sin(np.pi * j * h)) + (1 - w) * u[i, j] 

# Exact solution
exact_u = np.zeros((nodes + 1, nodes + 1))
for i in range(nodes + 1):
    for j in range(nodes + 1):
        exact_u[i, j] = (-1 / (2 * np.pi**2)) * np.sin(np.pi * i * h) * np.sin(np.pi * j * h)

# Error calculation
error = np.abs(u - exact_u)

# Plotting
x = np.linspace(0, 1, nodes + 1) 
y = np.linspace(0, 1, nodes + 1)
xx, yy = np.meshgrid(x, y)

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(xx, yy, u, cmap=plt.get_cmap('rainbow'))
ax1.set_title('Approximated Solution')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
ax1.text(0.1, 0.1, 0.8, f'h = {h}, w = {w}', color='black', fontsize=10)

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(xx, yy, exact_u, cmap=plt.get_cmap('rainbow'))
ax2.set_title('Exact Solution')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('u')
ax2.text(0.1, 0.1, 0.8, f'h = {h}, w = {w}', color='black', fontsize=10)

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(xx, yy, error, cmap=plt.get_cmap('rainbow'))
ax3.set_title('Error')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Error')
ax3.text(0.1, 0.1, 0.8, f'h = {h}, w = {w}', color='black', fontsize=10)

plt.tight_layout()
plt.show()




