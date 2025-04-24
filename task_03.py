import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic-Dataset-cleaned.csv")
#For Normalization
numerical_features = ['Age', 'Fare']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print("Normalized Dataset:")
print(df[numerical_features].head())

#For standardization
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print("Standardized Dataset:")
print(df[numerical_features].head())

