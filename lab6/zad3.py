import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('diabetes.csv')

all_inputs = df[
    ['pregnant-times', 'glucose-concentr', 'blood-pressure', 'skin-thickness', 'insulin', 'mass-index', 'pedigree-func',
     'age']].values
all_classes = df['class'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.67,
                                                                            random_state=1)
dtc = DecisionTreeClassifier()
dtcPred = dtc.fit(train_inputs, train_classes).predict(test_inputs)
print("Decision tree score:")
print(dtc.score(test_inputs, test_classes))
print("Decision tree confusion matrix:")
dtcCm = confusion_matrix(test_classes, dtcPred)
print(dtcCm)

knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn3.fit(train_inputs, train_classes)
knn3Pred = knn3.predict(test_inputs)
confusion_matrix(test_classes, knn3Pred)
print("\nKNN k=3 score:")
print(knn3.score(test_inputs, test_classes))
print("KNN k=3 confusion matrix:")
knn3Cm = confusion_matrix(test_classes, knn3Pred)
print(knn3Cm)

knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn5.fit(train_inputs, train_classes)
knn5Pred = knn5.predict(test_inputs)
confusion_matrix(test_classes, knn5Pred)
print("\nKNN k=5 score:")
print(knn5.score(test_inputs, test_classes))
print("KNN k=5 confusion matrix:")
knn5Cm = confusion_matrix(test_classes, knn5Pred)
print(knn5Cm)

knn11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn11.fit(train_inputs, train_classes)
knn11Pred = knn11.predict(test_inputs)
confusion_matrix(test_classes, knn11Pred)
print("\nKNN k=11 score:")
print(knn11.score(test_inputs, test_classes))
print("KNN k=11 confusion matrix:")
knn11Cm = confusion_matrix(test_classes, knn11Pred)
print(knn11Cm)

gnb = GaussianNB()
gnb.fit(train_inputs, train_classes)
gnbPred = gnb.predict(test_inputs)
confusion_matrix(test_classes, gnbPred)
print("\nGnb score:")
print(gnb.score(test_inputs, test_classes))
print("Gnb confusion matrix:")
gnbCm = confusion_matrix(test_classes, gnbPred)
print(gnbCm)

label = ['Drzewo', 'KNN k=3', 'KNN k=5', 'KNN k=11', 'Bayes']
results = [
    dtc.score(test_inputs, test_classes),
    knn3.score(test_inputs, test_classes),
    knn5.score(test_inputs, test_classes),
    knn11.score(test_inputs, test_classes),
    gnb.score(test_inputs, test_classes)
]

index = np.arange(len(label))
plt.bar(index, results)
plt.xlabel('Wynik', fontsize=5)
plt.ylabel('Metoda', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Porównanie skuteczności metod')
plt.show()
