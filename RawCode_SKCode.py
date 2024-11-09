import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
file_path = 'data.csv'  # Update this path to your dataset
data = pd.read_csv(file_path)

# Drop rows with any NaN values
data = data.dropna()

# Separate features and target
X = data.drop(columns=['Life Ladder'])  # Assuming 'Life Ladder' is the target column
y = data['Life Ladder']

# Handle categorical data: Use LabelEncoder for categorical columns
label_encoder = LabelEncoder()
categorical_columns = X.select_dtypes(include=['object']).columns

# Encode each categorical column using LabelEncoder
for col in categorical_columns:
    X[col] = label_encoder.fit_transform(X[col])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model (SKLearn)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions (SKLearn)
y_train_pred_sklearn = model.predict(X_train_scaled)
y_test_pred_sklearn = model.predict(X_test_scaled)

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
alpha = 0.01
iterations = 1000

# Add intercept term to the scaled training and testing sets
X_train_intercept = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_intercept = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# Run gradient descent
theta_final = gradient_descent(X_train_intercept, y_train.to_numpy(), theta_initial, alpha, iterations)

# Make predictions using Raw MLR
y_train_pred_manual = hypothesis(X_train_intercept, theta_final)
y_test_pred_manual = hypothesis(X_test_intercept, theta_final)

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
