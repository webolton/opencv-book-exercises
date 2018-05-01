# Linear decision bounderies

from sklearn import datasets

X, y = datasets.make_classification(n_samples=100, n_features=2,
                                    n_redundant=0, n_classes=2,
                                    random_state=7816)

X.shape, y.shape

# Plot dataset

import matplotlib.pyplot as plt
%matplotlib
plt.style.use('ggplot')
plt.set_cmap('jet')

plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.xlabel('x values')
plt.ylabel('y values')

# Process the dataset for OpenCV

import numpy as np
X = X.astype(np.float32)
y = y * 2 - 1

# Split the data into train and test sets

from sklearn import model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create the support vector machine
import cv2
svm = cv2.ml.SVM_create()

# Configure the SVM to try and partition the data with a straight line
svm.setKernel(cv2.ml.SVM_LINEAR)

# Call the train method on the SVM to find the optimal decision boundary
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train);

# Call the predict method on the SVM to predict the target lables in the test set
_, y_pred = svm.predict(X_test)

# Use sklearn metrics module to score classifier
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)

# Visualizing the decision boundry with a function

def plot_decision_boundary(svm, X_test, y_test):
    # create mesh to plot in
    h = 0.02 # step size in mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    X_hypo = np.c_[xx.ravel().astype(np.float32),
                   yy.ravel().astype(np.float32)]
    _, zz = svm.predict(X_hypo)
    zz = zz.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=200)

plt.figure(figsize=(10, 6))
plot_decision_boundary(svm, X_test, y_test)

# Non-linear decision bounderies

kernels = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF]

plt.figure(figsize=(14, 8))
for idx, kernel in enumerate(kernels):
    svm = cv2.ml.SVM_create()
    svm.setKernel(kernel)
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    _, y_pred = svm.predict(X_test)

    plt.subplot(2, 2, idx + 1)
    plot_decision_boundary(svm, X_test, y_test)
    plt.title('accuracy = %.2f' % metrics.accuracy_score(y_test, y_pred))
