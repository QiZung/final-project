import numpy as np
import matplotlib.pyplot as plt

# Define specific step sizes
step_sizes = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5]

# Initialize error list
errors = []

for h in step_sizes:
    # Define grid parameters
    nx = int(1 / h)  # Number of grid points in x direction
    ny = int(1 / h)  # Number of grid points in y direction
    dx = h  # Grid spacing in x direction
    dy = h  # Grid spacing in y direction

    # Define boundary conditions
    u_boundary = np.zeros((nx + 2, ny + 2))
    u_boundary[0, :] = 0  # Lower boundary
    u_boundary[-1, :] = 0  # Upper boundary
    u_boundary[:, 0] = 0  # Left boundary
    u_boundary[:, -1] = 0  # Right boundary

    # Initialize solution vector
    u = np.zeros((nx + 2, ny + 2))

    # Define the right-hand side
    f = np.zeros((nx + 2, ny + 2))
    x = np.linspace(0, 1, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    X, Y = np.meshgrid(x, y)
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)

    # Define the exact solution
    exact_u = np.zeros((nx + 2, ny + 2))
    for i in range(nx + 2):
        for j in range(ny + 2):
            exact_u[i, j] = (-1 / (2 * np.pi**2)) * np.sin(np.pi * i * dx) * np.sin(np.pi * j * dy)

    # Perform the iterative solution
    max_iter = 1000  # Maximum number of iterations
    tolerance = 1e-4  # Convergence tolerance

    for iteration in range(max_iter):
        u_old = u.copy()

        # Update the solution using the finite difference method
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                u[i, j] = (u_old[i+1, j] + u_old[i-1, j] + u_old[i, j+1] + u_old[i, j-1] - dx**2 * f[i, j]) / 4

        # Check convergence
        residual = np.max(np.abs(u - u_old))
        if residual < tolerance:
            break

    # Calculate the error
    error = np.abs(u[1:-1, 1:-1] - exact_u[1:-1, 1:-1])

    # Calculate the maximum error
    max_error = np.max(error)
    errors.append(max_error)

# Plot the maximum errors
plt.plot(step_sizes, errors, marker='o', label='Data')
plt.xlabel('Step Size (h)')
plt.ylabel('Maximum Error')
plt.title('Maximum Error vs. Step Size')
plt.legend()
plt.grid(True)
plt.show()