import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('iris2D.csv')
x = dataset.iloc[:, [1, 2]].values


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green')
plt.show()

X = StandardScaler().fit_transform(x)
db = DBSCAN(eps=0.2, metric='euclidean', min_samples=8).fit(X)
labels = db.labels_
plt.scatter(x[labels == 0, 0], x[labels == 0, 1], s=100, c='red')
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], s=100, c='blue')
plt.scatter(x[labels == 2, 0], x[labels == 2, 1], s=100, c='green')
plt.show()

x_pred = []

for x in range(150):
    if (x < 50):
        x_pred.append(0)
    if (50 <= x < 100):
        x_pred.append(1)
    if (100 <= x < 150):
        x_pred.append(2)

print(purity_score(x_pred, y_kmeans))
print(purity_score(x_pred, labels))
