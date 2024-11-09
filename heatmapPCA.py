import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv('data.csv')
data = data.dropna()

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

# Standardize the features (scaling)
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=8)  # You can adjust this to the number of components you need
numeric_data_pca = pca.fit_transform(numeric_data_scaled)

# Create a DataFrame for PCA components
pca_df = pd.DataFrame(numeric_data_pca, columns=[f'PCA Component {i+1}' for i in range(8)])

# Calculate the correlation matrix for the PCA components
corr_matrix_pca = pca_df.corr()

# Plotting the heatmap for PCA components' correlation
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_pca, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8})

# Customize the plot with titles and axis labels
plt.title('Correlation Heatmap of PCA Components', fontsize=16)
plt.xlabel('PCA Components', fontsize=12)
plt.ylabel('PCA Components', fontsize=12)

# Show the plot
plt.show()

# Explained Variance Ratio for each component
explained_variance = pca.explained_variance_ratio_

# Print the explained variance ratio for each component
print("Explained Variance Ratio for each PCA component:")
for i, variance in enumerate(explained_variance, 1):
    print(f"PCA Component {i}: {variance:.4f}")

# Print the cumulative explained variance
cumulative_variance = explained_variance.cumsum()
print("\nCumulative Explained Variance:")
for i, variance in enumerate(cumulative_variance, 1):
    print(f"PCA Component {i}: {variance:.4f}")

# PCA Component Loadings (The eigenvectors that show the contribution of each original feature to the components)
loadings = pca.components_

# Create a DataFrame to display the loadings
loading_df = pd.DataFrame(loadings.T, columns=[f'PCA Component {i+1}' for i in range(8)], index=numeric_data.columns)

# Print the loadings of the components
print("\nPCA Component Loadings:")
print(loading_df)

# Heatmap of PCA Loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loading_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8})

# Customize the plot with titles and axis labels for loadings
plt.title('Heatmap of PCA Component Loadings', fontsize=16)
plt.xlabel('PCA Components', fontsize=12)
plt.ylabel('Original Features', fontsize=12)

# Show the plot
plt.show()
