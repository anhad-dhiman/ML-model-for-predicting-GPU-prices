import math
import numpy as np
import matplotlib.pyplot as plt

# Load data
x_train = np.array([150, 200, 250, 300, 350, 400, 500])  # features -> GPU price
y_train = np.array([50, 65, 75, 90, 100, 110, 130])     # target -> FPS

# Normalize features to prevent overflow
x_mean = np.mean(x_train)
x_std = np.std(x_train)
x_norm = (x_train - x_mean) / x_std

# Compute cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    
    total_cost = (1 / (2 * m)) * cost
    return total_cost

def compute_derivative(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0   # partial derivative of cost function w.r.t w
    dj_db = 0   # partial derivative of cost function w.r.t b
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]  # partial derivative of ith feature w.r.t w
        dj_db_i = f_wb - y[i]           # partial derivative of ith feature w.r.t b
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    # final derivative
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the derivative for this step
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        
        # Print cost at intervals of 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    
    return w, b, J_history, p_history

# Initialize starting values
initial_w = 0
initial_b = 0
iterations = 10000
alpha = 0.01  #Learning rate for normalized data

# Train the model using normalized features
w_final, b_final, J_hist, p_hist = gradient_descent(x_norm, y_train, initial_w, initial_b,
                                                    alpha, iterations, compute_cost, compute_derivative)

print(f"\nFinal model (normalized): w = {w_final:.4f}, b = {b_final:.4f}")

# Convert back to original scale for predictions
w_original = w_final / x_std
b_original = b_final - (w_final * x_mean / x_std)

print(f"Final model (original scale): w = {w_original:.6f}, b = {b_original:.4f}")

# Prediction
gpu_price = 300
predicted_fps = w_original * gpu_price + b_original
print(f"Predicted FPS for ${gpu_price} GPU: {predicted_fps:.2f}")

# Plotting the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='red', label='Actual Data', s=50)
plt.plot(x_train, w_original * x_train + b_original, label='Regression Line', color='blue', linewidth=2)
plt.xlabel("GPU Price (USD $)")
plt.ylabel("Average FPS")
plt.title("GPU Price vs FPS - Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(J_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Over Iterations")
plt.grid(True, alpha=0.3)
plt.show()