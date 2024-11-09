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

# Adam Implementation
def adam(X, y, theta, alpha, iterations, epsilon=1e-8, beta1=0.9, beta2=0.999):
    m = len(y)
    cost_history = []
    m_t = np.zeros_like(theta)  # Initialize 1st moment estimate (mean of gradients)
    v_t = np.zeros_like(theta)  # Initialize 2nd moment estimate (variance of gradients)
    t = 0  # Time step
    
    for i in range(iterations):
        t += 1
        predictions = hypothesis(X, theta)
        error = predictions - y
        gradient = (1 / m) * np.dot(X.T, error)
        
        # Update biased first and second moment estimates
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * gradient ** 2
        
        # Correct bias in estimates
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)
        
        # Update parameters
        theta -= (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat
        
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
    return theta, cost_history

# Test Adam for different learning rates and store cost histories
cost_histories = {}
optimal_costs = {}
optimal_iterations = {}

for alpha in alpha_values:
    theta = theta_initial.copy()
    theta_final, cost_history = adam(X_intercept, y.to_numpy(), theta, alpha, iterations)
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
plt.title('Cost History for Adam Optimizer with Different Learning Rates')
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
