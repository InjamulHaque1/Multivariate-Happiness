import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('data.csv')

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

# Calculate the correlation matrix for numeric columns (now including label encoded columns)
corr_matrix = numeric_data.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8})

# Customize the plot with titles and axis labels
plt.title('Correlation Heatmap of Features with Label Encoding', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Show the plot
plt.show()
