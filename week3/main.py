import cv2
import numpy as np
import os

os.makedirs('results', exist_ok=True)

folderInput = 'images/'
figure_names =["circles.png","noisedCircles.tif","phantom1.bmp","phantom2.bmp","phantom3.bmp"]
figure_name = folderInput + figure_names[0]

img = cv2.imread(figure_name, cv2.IMREAD_GRAYSCALE)
img = img.astype('float')

# grayscale image
cv2.imwrite('results/01_grayscale.png', img)

# Normalize image
img = (img.astype('float') - np.min(img))
img = img/np.max(img)
cv2.imwrite('results/02_normalized.png',img)

ni = img.shape[0]
nj = img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
# if len(img.shape) > 2:
#     nc = img.shape[2] # number of channels
#     img = np.mean(img, axis=2)

# Try out different parameters
mu = 1
nu = 1
lambda1 = 1
lambda2 = 1
tol = 0.1
dt = (1e-2)/mu
iterMax = 1e5

X, Y = np.meshgrid(np.arange(0, nj), np.arange(0, ni), indexing='xy')

# Initial phi
# This initialization allows a faster convergence for phantom2
phi = (-np.sqrt((X - np.round(ni / 2)) ** 2 + (Y - np.round(nj / 2)) ** 2) + 50)
# Alternatively, you may initialize randomly, or use the checkerboard pattern as suggested in Getreuer's paper

# Normalization of the initial phi to the range [-1, 1]
min_val = np.min(phi)
max_val = np.max(phi)
phi = phi - min_val
phi = 2 * phi / max_val
phi = phi - 1

# CODE TO COMPLETE
# Explicit gradient descent or Semi-explicit (Gauss-Seidel) gradient descent (Bonus)
# Extra: Implement the Chan-Sandberg-Vese model (for colored images)
# Refer to Getreuer's paper (2012)

# Segmented image
seg = np.zeros(shape=img.shape)

# CODE TO COMPLETE

# Show output image
cv2.imshow('Segmented image', seg); cv2.waitKey(0)