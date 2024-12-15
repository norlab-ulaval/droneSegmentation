import os
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('/home/kamyar/Documents/M2F_Results/MV/2023-06-29-131242-5-Lac-Saint-Jean-4000x4000-DJI-FC7303-patch-4.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
