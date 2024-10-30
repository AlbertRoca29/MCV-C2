import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

os.makedirs('results', exist_ok=True)

folderInput = 'images/'
figure_names =["circles.png","noisedCircles.tif","phantom1.bmp","phantom2.bmp","phantom3.bmp"]
figure_name = folderInput + figure_names[2]

img = cv2.imread(figure_name, cv2.IMREAD_GRAYSCALE)
img = img.astype('float')
cv2.imwrite('results/01_grayscale.png', img)

img = (img.astype('float') - np.min(img))/ np.max(img)
cv2.imwrite('results/02_normalized.png',img *255)

img = img

ni, nj = img.shape[0], img.shape[1]

# Make color images grayscale. Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
if len(img.shape) > 2:
    nc = img.shape[2] # number of channels
    img = np.mean(img, axis=2)

# Parameters (try different)
mu = 1
nu = .1
lambda1 = 1
lambda2 = 1
tol = 1e-4
dt = 1
max_iter  = 100

X, Y = np.meshgrid(np.arange(nj), np.arange(ni))

# phi = -np.sqrt((X - nj / 2)**2 + (Y - ni / 2)**2) + 30
phi = np.sin( np.pi/5 * X)*np.sin(np.pi/5 * Y)

# Helper functions
def heaviside(x, epsilon=1.0):
    return 0.5 * (1 + (2 / np.pi) * np.arctan(x / epsilon))

def dirac(x, epsilon=1.0):
    return epsilon / (np.pi * (epsilon**2 + x**2))

c1 = 0
c2 = 1

# Explicit gradient descent loop
for iteration in range(max_iter):
    # Calculate region averages c1 and c2
    H_phi = heaviside(phi)
    
    # Compute forward and backward differences
    grad_x_fwd = np.roll(phi, -1, axis=1) - phi
    grad_x_bwd = phi - np.roll(phi, 1, axis=1)
    grad_y_fwd = np.roll(phi, -1, axis=0) - phi
    grad_y_bwd = phi - np.roll(phi, 1, axis=0)
    grad_y_centered = (grad_y_fwd + grad_y_bwd) / 2
    grad_x_centered = (grad_x_fwd + grad_x_bwd) / 2 

    # Calculate curvature terms for explicit gradient descent
    A = mu / np.sqrt(1e-8 + grad_x_fwd**2 + grad_y_centered**2)
    B = mu / np.sqrt(1e-8 + grad_y_fwd**2 + grad_x_centered**2)

    # Update phi explicitly
    d_phi = dirac(phi)
    curvature_term = (A * np.roll(phi, -1, axis=1) + A * np.roll(phi, 1, axis=1)
                     + B * np.roll(phi, -1, axis=0) + B * np.roll(phi, 1, axis=0) )
    
    data_term = - lambda1 * (img - c1)**2 + lambda2 * (img - c2)**2

    phi = (phi + dt * d_phi * (curvature_term - nu + data_term)) / (1 +dt)

    # Check for convergence
    print(iteration)
    plt.show()
    if iteration > 0 and np.linalg.norm(phi - phi_prev) / phi.size < tol:
        break
    phi_prev = phi.copy()

# Display final segmented image
segmentation = phi >= 0
plt.imshow(segmentation, cmap='gray')
plt.title("Segmented Image")
plt.show()

# Segmented image
# seg = np.zeros(shape=img.shape)

# CODE TO COMPLETE

# Show output image
cv2.imwrite('results/09_segmented_image.png', np.array(segmentation).astype(float)*255)