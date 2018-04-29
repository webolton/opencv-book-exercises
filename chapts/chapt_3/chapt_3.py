import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib

plt.style.use('ggplot')

np.random.seed(42)

single_data_point = np.random.randint(0, 100, 2)

single_data_point

single_label = np.random.randint(0, 2)

def generate_data(num_samples, num_features=2):
    """Randomly generates a number of data points"""
    data_size = (num_samples, num_features)
    train_data = np.random.randint(0, 100, size=data_size)
    labels_size = (num_samples, 1)
    labels = np.random.randint(0, 2, size=labels_size)

    return train_data.astype(np.float32), labels

def plot_data(all_blue, all_red):
    plt.figure(figsize=(10, 6))
    plt.scatter(all_blue[:, 0], all_blue[:, 1], c='b', marker='s', s=180)
    plt.scatter(all_red[:, 0], all_red[:, 1], c='r', marker='^', s=180)
    plt.xlabel('x coordinate (feature 1)')
    plt.ylable('y coordinate (feature 2)')

