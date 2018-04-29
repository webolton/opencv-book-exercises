import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model
%matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

boston = datasets.load_boston()
dir(boston)

linreg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = modsel.train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=42
)

linreg.fit(x_train, y_train)
