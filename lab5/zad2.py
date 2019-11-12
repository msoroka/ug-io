import pandas as pd
import pydot

from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('iris.csv')

all_inputs = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']].values
all_classes = df['class'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7,
                                                                            random_state=1)
dtc = DecisionTreeClassifier()
pred = dtc.fit(train_inputs, train_classes).predict(test_inputs)

dot_data = StringIO()
tree.export_graphviz(dtc, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())

graph[0].write_pdf("iris.pdf")

print("Score:")
print(dtc.score(test_inputs, test_classes))

print("\nConfusion matrix:")
cm = confusion_matrix(test_classes, pred)
print(cm)
