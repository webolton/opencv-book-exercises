import matplotlib.pyplot as plt
%matplotlib
plt.style.use('ggplot')

data = [
    {'age': 33, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.66, 'K': 0.06, 'drug': 'A'},
    {'age': 77, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.19, 'K': 0.03, 'drug': 'D'},
    {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.80, 'K': 0.05, 'drug': 'B'},
    {'age': 39, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.19, 'K': 0.02, 'drug': 'C'},
    {'age': 43, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'high', 'Na': 0.36, 'K': 0.03, 'drug': 'D'},
    {'age': 82, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.09, 'K': 0.09, 'drug': 'C'},
    {'age': 40, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.89, 'K': 0.02, 'drug': 'A'},
    {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.80, 'K': 0.05, 'drug': 'B'},
    {'age': 29, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.35, 'K': 0.04, 'drug': 'D'},
    {'age': 53, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.54, 'K': 0.06, 'drug': 'C'},
    {'age': 36, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.53, 'K': 0.05, 'drug': 'A'},
    {'age': 63, 'sex': 'M', 'BP': 'low', 'cholesterol': 'high', 'Na': 0.86, 'K': 0.09, 'drug': 'B'},
    {'age': 60, 'sex': 'M', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.66, 'K': 0.04, 'drug': 'C'},
    {'age': 55, 'sex': 'M', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.82, 'K': 0.04, 'drug': 'B'},
    {'age': 35, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'high', 'Na': 0.27, 'K': 0.03, 'drug': 'D'},
    {'age': 23, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.55, 'K': 0.08, 'drug': 'A'},
    {'age': 49, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.27, 'K': 0.05, 'drug': 'C'},
    {'age': 27, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.77, 'K': 0.02, 'drug': 'B'},
    {'age': 51, 'sex': 'F', 'BP': 'low', 'cholesterol': 'high', 'Na': 0.20, 'K': 0.02, 'drug': 'D'},
    {'age': 38, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.78, 'K': 0.05, 'drug': 'A'}
]

target = [d['drug'] for d in data]
target
[d.pop('drug') for d in data];

age = [d['age'] for d in data]
age

sodium = [d['Na'] for d in data]
potassium = [d['K'] for d in data]

plt.scatter(sodium, potassium)
plt.xlabel('sodium')
plt.ylabel('potassium')

target = [ord(t) - 65 for t in target]
target

plt.figure(figsize=(14, 10))

plt.subplot(221)
plt.scatter(sodium, potassium, c=target, s=100)
plt.xlabel('sodium (Na')
plt.ylabel('potassium (K)')
plt.subplot(222)
plt.scatter(age, potassium, c=target, s=100)
plt.xlabel('age')
plt.ylabel('potassium (K)')
plt.subplot(223)
plt.scatter(age, sodium, c=target, s=100)
plt.xlabel('age')
plt.ylabel('sodium (Na)')

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
data_pre = vec.fit_transform(data)
vec.get_feature_names()
data_pre[0]

import numpy as np
data_pre = np.array(data_pre, dtype=np.float32)
target = np.array(target, dtype=np.float32)

import sklearn.model_selection as ms
x_train, x_test, y_train, y_test = ms.train_test_split(data_pre, target, test_size=5,
    random_state=42)

# import cv2
# dtree = cv2.ml.DTrees_create()
# dtree.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
# y_pred = dtree.predict(x_test)
# from sklearn import metrics
# metrics.accuracy_score(y_test, dtree.predict(x_test))

from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(x_train, y_train)

dtc.score(x_train, y_train)
dtc.score(x_test, y_test)

with open('tree.dot', 'w')
f = tree.export_graphviz(clf, out_file=f)



