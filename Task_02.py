#By one hot encoding

import pandas as pd

df = pd.read_csv("Titanic-Dataset-cleaned.csv")

for col in df.columns:
    if df[col].dtype == 'int64' and df[col].nunique() < 10:
        df[col] = df[col].astype('object')

# Load your dataset


categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(df_encoded.dtypes[df_encoded.dtypes == 'object'])
print(df_encoded.head())


print("Before encoding:", df.shape)
print("After encoding :", df_encoded.shape)
