import numpy as np
import pandas as pd

missing_values = ["n/a", "na", "N/A", "NA", "--", "-"]
df = pd.read_csv('iris_with_errors.csv', na_values=missing_values)
print(df.isnull().sum())

medianSL = df['sepal.length'].median()
medianSW = df['sepal.width'].median()
medianPL = df['petal.length'].median()
medianPW = df['petal.width'].median()

num = df._get_numeric_data()

num['sepal.length'][num['sepal.length'] < 0] = medianSL
num['sepal.length'][num['sepal.length'] > 15] = medianSL
num['sepal.width'][num['sepal.width'] < 0] = medianSW
num['sepal.width'][num['sepal.width'] > 15] = medianSW
num['petal.length'][num['petal.length'] < 0] = medianPL
num['petal.length'][num['petal.length'] > 15] = medianPL
num['petal.width'][num['petal.width'] < 0] = medianPW
num['petal.width'][num['petal.width'] > 15] = medianPW

df['sepal.length'].fillna(medianSL, inplace=True)
df['sepal.width'].fillna(medianSW, inplace=True)
df['petal.length'].fillna(medianPL, inplace=True)
df['petal.width'].fillna(medianPW, inplace=True)

for key, row in df.iterrows():
    if 0 <= key <= 49:
        df.loc[key, 'variety'] = 'Setosa'
    if 50 <= key <= 99:
        df.loc[key, 'variety'] = 'Versicolor'
    if 100 <= key <= 149:
        df.loc[key, 'variety'] = 'Virginica'

df.to_csv('iris_without_errors.csv')
