import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
file_path = 'data.csv'  # Update this path to your dataset
data = pd.read_csv(file_path)

# Drop rows with any NaN values
data = data.dropna()

# Separate features and target
X = data.drop(columns=['Life Ladder'])  # Assuming 'Life Ladder' is the target column
y = data['Life Ladder']

# Convert categorical columns using one-hot encoding
X = pd.get_dummies(X)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term (for the bias term in linear regression)
X_intercept = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# Initialize parameters
alpha_values = [0.001, 0.01, 0.05]
iterations = 1000
theta_initial = np.zeros(X_intercept.shape[1])
convergence_threshold = 1e-6  # Convergence threshold to stop early if cost stops changing

# Hypothesis and cost function definitions
def hypothesis(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Gradient Descent Implementation
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = hypothesis(X, theta)
        error = predictions - y
        gradient = (1 / m) * np.dot(X.T, error)
        
        # Update parameters
        theta -= alpha * gradient
        
        # Compute and store cost for current iteration
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        # Early stopping based on convergence threshold
        if i > 0 and abs(cost_history[i] - cost_history[i-1]) < convergence_threshold:
            print(f"Converged at iteration {i}")
            break
    
    return theta, cost_history

# Test Gradient Descent for different learning rates and store cost histories
cost_histories = {}
optimal_costs = {}
optimal_iterations = {}

for alpha in alpha_values:
    theta = theta_initial.copy()
    theta_final, cost_history = gradient_descent(X_intercept, y.to_numpy(), theta, alpha, iterations)
    cost_histories[alpha] = cost_history
    min_cost = min(cost_history)
    min_iteration = cost_history.index(min_cost)
    optimal_costs[alpha] = min_cost
    optimal_iterations[alpha] = min_iteration

# Plotting cost histories for different learning rates (Alpha values)
plt.figure(figsize=(12, 8))

# Plot the cost history for each alpha
for alpha in alpha_values:
    plt.plot(cost_histories[alpha], label=f'Alpha = {alpha}', linewidth=2)

# Highlight optimal points for each alpha
for alpha in alpha_values:
    min_cost = optimal_costs[alpha]
    min_iteration = optimal_iterations[alpha]
    plt.scatter(min_iteration, min_cost, color='red', s=100, edgecolor='black')
    plt.text(min_iteration, min_cost, f"Cost = {min_cost:.4f}\nIter = {min_iteration}", 
             fontsize=12, color='red', ha='center', va='bottom')

# Plot settings
plt.title('Cost History for Gradient Descent with Different Learning Rates')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Display the best alpha (lowest cost) and optimal iteration
best_alpha = min(optimal_costs, key=optimal_costs.get)
best_iteration = optimal_iterations[best_alpha]
best_cost = optimal_costs[best_alpha]

print(f"Best Learning Rate (Alpha): {best_alpha}")
print(f"Optimal Iteration: {best_iteration} with Minimum Cost: {best_cost}")
