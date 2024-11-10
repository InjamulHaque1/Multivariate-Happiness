import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
# -------------------- PART 1: Preprocessing and Heatmap --------------------

# Load dataset
data = pd.read_csv('data.csv')
data = data.dropna()  # Drop rows with any NaN values

# Fill missing values in categorical columns with 'Missing'
categorical_columns = data.select_dtypes(include=['object']).columns  # Dynamically select categorical columns
data[categorical_columns] = data[categorical_columns].fillna('Missing')

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to each categorical column
for column in categorical_columns:
    if data[column].dtype == 'object':  # Only apply label encoding to categorical columns
        data[column] = label_encoder.fit_transform(data[column])

# Select only numeric columns (to calculate correlation matrix)
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns (now including label-encoded columns)
corr_matrix = numeric_data.corr()

# Plotting the heatmap
plt.figure(figsize=(14, 10))  # Increase the figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8})

# Customize the plot with titles and axis labels
plt.title('Correlation Heatmap of Features with Label Encoding', fontsize=16)

# Rotate and adjust labels
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)  # Keep y-axis labels horizontal

# Add padding and layout adjustments
plt.tight_layout(pad=2.0)  # Add padding around the plot

# Show the plot
plt.show()

# -------------------- PART 2: Linear Regression vs. Adam Optimization --------------------
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

# Initialize parameters and find optimal Alpha
alpha_values = [0.001, 0.01]
iterations = 1000
theta_initial = np.zeros(X_intercept.shape[1])

# Hypothesis and cost function definitions for Adam
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

# -------------------- PART 3: Linear Regression with SKLearn --------------------

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply the same scaling to X_test

# Initialize and train the model (SKLearn)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions (SKLearn)
start_time = time.time()
y_train_pred_sklearn = model.predict(X_train_scaled)
y_test_pred_sklearn = model.predict(X_test_scaled)  # Ensure that predictions are made on the scaled X_test
sklearn_time = time.time() - start_time

# -------------------- PART 4: Raw MLR (Manual Implementation) --------------------

# Raw MLR (Manual Implementation)
def hypothesis(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = hypothesis(X, theta)
        error = predictions - y
        gradient = (1 / m) * np.dot(error, X)
        theta -= alpha * gradient
    return theta

# Initialize parameters for Raw MLR
theta_initial = np.zeros(X_train_scaled.shape[1] + 1)  # +1 for the intercept term
alpha = best_alpha
iterations = best_iteration

# Add intercept term to the scaled training and testing sets
X_train_intercept = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_intercept = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Run gradient descent
start_time = time.time()
theta_final = gradient_descent(X_train_intercept, y_train.to_numpy(), theta_initial, alpha, iterations)
manual_time = time.time() - start_time 

# Make predictions using Raw MLR
y_train_pred_manual = hypothesis(X_train_intercept, theta_final)
y_test_pred_manual = hypothesis(X_test_intercept, theta_final)

# -------------------- PART 5: Plot Results (Training and Testing Data vs Prediction) --------------------

# Plot results (Training and Testing Data vs Prediction)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SK-Learn Training vs Prediction
axes[0, 0].scatter(y_train, y_train_pred_sklearn, color='green', alpha=0.6)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_title('SK-Learn MLR: Training Data vs Prediction')
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predicted Values')

# SK-Learn Testing vs Prediction
axes[0, 1].scatter(y_test, y_test_pred_sklearn, color='green', alpha=0.6)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_title('SK-Learn MLR: Testing Data vs Prediction')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predicted Values')

# Raw MLR Training vs Prediction (Manual Implementation)
axes[1, 0].scatter(y_train, y_train_pred_manual, color='blue', alpha=0.6)
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 0].set_title('Raw MLR: Training Data vs Prediction')
axes[1, 0].set_xlabel('True Values')
axes[1, 0].set_ylabel('Predicted Values')

# Raw MLR Testing vs Prediction (Manual Implementation)
axes[1, 1].scatter(y_test, y_test_pred_manual, color='blue', alpha=0.6)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_title('Raw MLR: Testing Data vs Prediction')
axes[1, 1].set_xlabel('True Values')
axes[1, 1].set_ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# ------------------------------ PART 6: Evaluation Matrcs ---------------------------

# Evaluate SKLearn model on training data
mae_sklearn_train = mean_absolute_error(y_train, y_train_pred_sklearn)
mse_sklearn_train = mean_squared_error(y_train, y_train_pred_sklearn)
rmse_sklearn_train = np.sqrt(mse_sklearn_train)
r2_sklearn_train = r2_score(y_train, y_train_pred_sklearn)

# Evaluate SKLearn model on testing data
mae_sklearn_test = mean_absolute_error(y_test, y_test_pred_sklearn)
mse_sklearn_test = mean_squared_error(y_test, y_test_pred_sklearn)
rmse_sklearn_test = np.sqrt(mse_sklearn_test)
r2_sklearn_test = r2_score(y_test, y_test_pred_sklearn)

# Evaluate Raw MLR model on training data
mae_manual_train = mean_absolute_error(y_train, y_train_pred_manual)
mse_manual_train = mean_squared_error(y_train, y_train_pred_manual)
rmse_manual_train = np.sqrt(mse_manual_train)
r2_manual_train = r2_score(y_train, y_train_pred_manual)

# Evaluate Raw MLR model on testing data
mae_manual_test = mean_absolute_error(y_test, y_test_pred_manual)
mse_manual_test = mean_squared_error(y_test, y_test_pred_manual)
rmse_manual_test = np.sqrt(mse_manual_test)
r2_manual_test = r2_score(y_test, y_test_pred_manual)

# Evaluation metrics for both models (training and testing)
metrics = ['MAE', 'MSE', 'RMSE', 'R²']

# Values for SKLearn (train and test)
sklearn_train_values = [mae_sklearn_train, mse_sklearn_train, rmse_sklearn_train, r2_sklearn_train]
sklearn_test_values = [mae_sklearn_test, mse_sklearn_test, rmse_sklearn_test, r2_sklearn_test]

# Values for Raw MLR (train and test)
manual_train_values = [mae_manual_train, mse_manual_train, rmse_manual_train, r2_manual_train]
manual_test_values = [mae_manual_test, mse_manual_test, rmse_manual_test, r2_manual_test]

# Set up the bar chart
x = np.arange(len(metrics))  # The label locations
width = 0.2  # The width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for both models on training and testing data
rects1_train = ax.bar(x - width, sklearn_train_values, width, label='SKLearn Train', color='skyblue')
rects2_train = ax.bar(x, manual_train_values, width, label='Raw MLR Train', color='indianred')
rects1_test = ax.bar(x + width, sklearn_test_values, width, label='SKLearn Test', color='deepskyblue')
rects2_test = ax.bar(x + 2 * width, manual_test_values, width, label='Raw MLR Test', color='lightcoral')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_title('Training vs Testing Evaluation Metrics for SKLearn and Raw MLR')
ax.set_xticks(x + width / 2)  # Center the ticks between train and test bars
ax.set_xticklabels(metrics)
ax.legend()

# Label the bars with the numeric values
def label_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}' if height != sklearn_time and height != manual_time else f'{height:.4f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call the function to add labels to bars
label_bars(rects1_train)
label_bars(rects2_train)
label_bars(rects1_test)
label_bars(rects2_test)

# Show the plot
plt.tight_layout()
plt.show()
