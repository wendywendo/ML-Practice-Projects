# GRADIENT DESCENT
# Developing a program to find the optimal parameters that minimize our loss function.
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Sample dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])


# Initialize parameters
beta_0 = 0.0 # intercept
beta_1 = 0.0 # slope
learning_rate = 0.01
iterations = 1000

n = len(X)

# Store parameters and loss at each iteration for animation
beta_0_list = []
beta_1_list = []
loss_list = []


# Gradient descent loop
for i in range(iterations):
    # Predictions
    y_pred = beta_0 + beta_1 * X
    loss = np.sum((y - y_pred) ** 2)

    # Compute gradients
    d_beta_0 = (-2/n) * np.sum(y - y_pred)
    d_beta_1 = (-2/n) * np.sum(X * (y - y_pred))

    # Update parameters
    beta_0 = beta_0 - learning_rate * d_beta_0
    beta_1 = beta_1 - learning_rate * d_beta_1

    beta_0_list.append(beta_0)
    beta_1_list.append(beta_1)
    loss_list.append(loss)


# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Sublot 1: Data and regression line
ax1.scatter(X, y, color='blue', label='Data points')
line, = ax1.plot([], [], color='red', label='Regression line')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 12)
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Gradient Descent Regression')
ax1.legend()


# Subplot 2: Loss over iterations
ax2.set_xlim(0, iterations)
ax2.set_ylim(0, max(loss_list)+1)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Decrease')
loss_line, = ax2.plot([], [], color='green', label='Loss')
ax2.legend()


# Animation function
def animate(i):
    # Compute the predicted values (regression line)
    y_pred = beta_0_list[i] + beta_1_list[i] * X
    line.set_data(X, y_pred)

    # Update loss line
    loss_line.set_data(range(i + 1), loss_list[:i+1])
    
    return line, loss_line

# Create animation
ani = FuncAnimation(fig, animate, frames=iterations, interval=100, blit=True)

plt.tight_layout()
plt.show()
