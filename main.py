import os
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('/home/kamyar/Documents/M2F_Results/MV/2024-06-19-120137-13.075-ZecChapais-5280x5280-DJI-M3E-patch-3.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
