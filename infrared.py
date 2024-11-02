import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load the image
image = cv2.imread("/home/med/Desktop/datasets/check_img/siickness.png")

cv2.imshow('originale', image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

infrared_image = cv2.merge([gray_image, gray_image, gray_image])
infrared_image[:, :, 0] = np.clip(infrared_image[:, :, 0] * 0.5, 0, 255)
infrared_image[:, :, 1] = np.clip(infrared_image[:, :, 1] * 0.5, 0, 255)

# Adjust the contrast and brightness to simulate the infrared effect
alpha = 1.5  # Contrast control
beta = 20    # Brightness control
infrared_image = cv2.convertScaleAbs(infrared_image, alpha=alpha, beta=beta)

# Save the result
cv2.imshow("infrared_image", infrared_image)

# Apply Canny edge detection
edges = cv2.Canny(gray_image, 100, 200)  # Adjust threshold values as needed

infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_BGR2RGB)

# Redimensionner l'image si nécessaire
(h, w) = infrared_image.shape[:2]
image = cv2.resize(infrared_image, (w // 1, h // 1))

# Reshaper l'image en un vecteur de pixels
pixels = image.reshape((-1, 3))

# Appliquer K-means
k = 8  # nombre de clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(pixels)

# Obtenir les labels et les couleurs des clusters
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Reshaper les labels en l'image segmentée
segmented_image = centers[labels].reshape(image.shape).astype(np.uint8)

# Afficher l'image originale et l'image segmentée
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Image Originale')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('Image Segmentée')
plt.imshow(segmented_image)
plt.show()

# Initialiser un dictionnaire pour stocker les comptes de chaque couleur
color_counts = defaultdict(int)
# Save the result
#cv2.imshow('infrared image', edges)

cv2.waitKey(0)