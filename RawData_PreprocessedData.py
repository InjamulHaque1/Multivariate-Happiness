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
numerical_cols = ['Life Ladder', 'Log GDP Per Capita', 'Social Support', 'Healthy Life Expectancy At Birth', 
                  'Freedom To Make Life Choices', 'Generosity', 'Perceptions Of Corruption', 
                  'Positive Affect', 'Negative Affect', 'Confidence In National Government']

# Copy the data and apply scaling
data_scaled = data.copy()
data_scaled[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Plotting box plots for each numerical column before and after scaling
for col in numerical_cols:
    plt.figure(figsize=(12, 6))
    
    # Before scaling - Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data[col], color='blue')
    plt.title(f'{col} - Raw Data Boxplot')
    plt.xlabel(col)
    plt.ylabel('Value')
    
    # After scaling - Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=data_scaled[col], color='red')
    plt.title(f'{col} - Scaled Data Boxplot')
    plt.xlabel(col)
    plt.ylabel('Value')
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()
