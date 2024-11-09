import pandas as pd

# Load dataset
file_path = 'data.csv'  # Update this path to your dataset
data = pd.read_csv(file_path)

# Initial inspection
print("First 5 rows of the dataset:")
print(data.head())  # View the first few rows

print("\nDataset Info:")
print(data.info())  # Column names, data types, non-null counts

# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isna().sum())  # Count missing values per column

# Drop rows with missing target values and confirm shape
data = data.dropna(subset=['Life Ladder'])
print(f"\nShape of the dataset after dropping rows with missing target values: {data.shape}")

# Separate features and target
X = data.drop(columns=['Life Ladder'])
y = data['Life Ladder']

# Print categorical and numerical features
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
print(f"\nCategorical Columns: {list(categorical_columns)}")
print(f"Numerical Columns: {list(numerical_columns)}")

# Descriptive statistics for numerical features
print("\nDescriptive Statistics for Numerical Features:")
print(X[numerical_columns].describe())

# Distribution of target variable
print("\nDistribution of Target Variable:")
print(y.describe())  # Summary stats for target
print("Unique values in target:", y.unique())  # Unique values if target is categorical

# Example of unique values in categorical columns
print("\nUnique Values in Categorical Columns:")
for col in categorical_columns:
    print(f"{col}: {X[col].unique()}")  # Unique values for each categorical feature
