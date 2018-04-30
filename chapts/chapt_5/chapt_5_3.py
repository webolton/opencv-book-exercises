from sklearn import datasets

data = datasets.load_breast_cancer()
data.data.shape

data.feature_names
data.target_names

import sklearn.model_selection as ms
x_train, x_test, y_train, y_test = ms.train_test_split(data.data, data.target, test_size=0.2, random_state=42)

x_train.shape, x_test.shape

from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(x_train, y_train)

with open("tree.dot", "w") as f:
    f = tree.export_graphviz(dtc, out_file=f,
                             feature_names=data.feature_names,
                             class_names=data.target_names)