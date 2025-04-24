import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Titanic-Dataset-cleaned.csv")

# Numerical features to analyze
numerical_features = ['Age', 'Fare']

# Visualize outliers using boxplots
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=feature)
    plt.title(f"Boxplot of {feature}")
    plt.show()

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # 25th percentile
    Q3 = df[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1                   # Interquartile range
    lower_bound = Q1 - 1.5 * IQR    # Lower bound
    upper_bound = Q3 + 1.5 * IQR    # Upper bound
    
    # Filter the DataFrame
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

# Remove outliers for numerical features
for feature in numerical_features:
    df = remove_outliers(df, feature)

# Display the cleaned dataset
print("Dataset after removing outliers:")
print(df.head())
