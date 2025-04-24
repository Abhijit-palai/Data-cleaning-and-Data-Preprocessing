import pandas as pd
from sklearn.impute import SimpleImputer

df=pd.read_csv("Titanic-Dataset.csv")
print(df.head())

print (df.info())

print(df.isnull().sum())

df['Age']=df['Age'].fillna(df['Age'].mean())
df['Cabin'] = df['Cabin'].fillna('Unknown')

#df['Embarked']=df['Embarked'].fillna(df['Embarked'].mean())

data_imputer = SimpleImputer(strategy='most_frequent')
df[['Embarked']] = data_imputer.fit_transform(df[['Embarked']])

df.to_csv('Titanic-Dataset-cleaned.csv',index=False)
print("clean data successfully")


#Convert categorical features into numerical using encoding.
# Convert any column with few unique int values to 'object' type

