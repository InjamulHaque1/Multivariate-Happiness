import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load your actual dataset (replace this if you're using a different format)
data = pd.read_csv("data.csv")

# Drop or encode categorical columns (we're dropping 'Country Name' and 'Regional Indicator')
data = data.drop(columns=['Country Name', 'Regional Indicator'])

# Check for missing values and fill with mean for numerical columns
data = data.fillna(data.mean())

# Scaling the numerical features
scaler = StandardScaler()

# List of numerical columns to scale
numerical_cols = [
    'Life Ladder', 'Log GDP Per Capita', 'Social Support', 
    'Healthy Life Expectancy At Birth', 'Freedom To Make Life Choices', 
    'Generosity', 'Perceptions Of Corruption', 'Positive Affect', 
    'Negative Affect', 'Confidence In National Government'
]

# Apply scaling
data_scaled = data.copy()
data_scaled[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Plotting scaled box plots with 3 columns per figure
num_plots = len(numerical_cols)
plots_per_fig = 3
num_figures = (num_plots + plots_per_fig - 1) // plots_per_fig  # Calculate the number of figures needed

for fig_num in range(num_figures):
    plt.figure(figsize=(18, 6))  # Adjust figure size for three plots side-by-side
    for i in range(plots_per_fig):
        col_idx = fig_num * plots_per_fig + i
        if col_idx >= num_plots:
            break
        col = numerical_cols[col_idx]
        
        # Boxplot for scaled data only
        plt.subplot(1, plots_per_fig, i + 1)
        sns.boxplot(data=data_scaled[col], color='red')
        plt.title(f'{col} - Scaled Data')
        plt.xlabel(col)
        plt.ylabel('Value')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
