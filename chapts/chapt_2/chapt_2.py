import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
%matplotlib

digits = datasets.load_digits()

print(digits.data.shape)
print(digits.images.shape)

img = digits.images[0, :, :]

plt.imshow(img, cmap='gray')

for image_index in range(10):
    # images are 0-indexed, but suplots are 1-indexed
    subplot_index = image_index + 1
    plt.subplot(2, 5, subplot_index)
    plt.imshow(digits.images[image_index, :, :], cmap='gray')

