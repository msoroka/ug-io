import numpy as np
import pandas as pd

df = pd.read_csv('iris.csv')


def myPredictRow(sl, sw, pl, pw):
    if pw < 1:
        return ('Iris-setosa')
    else:
        if 4.9 <= sl <= 7.0 and 2.0 <= sw <= 3.4 and 3.0 <= pl <= 5.1 and 1.0 <= pw <= 1.8 and 2.0 <= sw * pw <= 5.8 and 0.9 <= sl - pl <= 2.3:
            return ('Iris-versicolor')
        else:
            return ('Iris-virginica')


def myPredict(predict, real, row):
    if predict == real:
        return True


predict = 0
for key, row in df.iterrows():
    if (
            myPredict(myPredictRow(row['sepallength'], row['sepalwidth'], row['petallength'], row['petalwidth']),
                      row['class'],
                      row)):
        predict += 1

print(150 - predict)
print((predict * 100) / 150)
