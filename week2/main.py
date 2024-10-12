# %%
import cv2
import numpy as np
import poisson_editing
from scipy.sparse.linalg import spsolve
# %%
# Load images
src = cv2.imread('images/lena/girl.png')
dst = cv2.imread('images/lena/lena.png')
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.

# %%
# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
cv2.imshow('Source image', src); cv2.waitKey(0)
cv2.imshow('Destination image', dst); cv2.waitKey(0)

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

# %%
# Get the translation vectors 
# t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
# t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

# Cut out the relevant parts from the source image and shift them into the right position

new_src = np.zeros_like(src)
ind_eyes = np.argwhere(src_mask_eyes[:,:,0] > 0)
new_src[ind_eyes[:, 0]+t_eyes[0], ind_eyes[:, 1]+t_eyes[1]] = src[ind_eyes[:, 0], ind_eyes[:, 1]]

ind_mouth = np.argwhere(src_mask_mouth[:,:,0] > 0) 
new_src[ind_mouth[:, 0]+ t_mouth[0], ind_mouth[:, 1]+ t_mouth[1]] = src[ind_mouth[:, 0], ind_mouth[:, 1]]

cv2.imshow('Enganxat!', new_src); cv2.waitKey(0)

# %%
u_comb = np.zeros_like(dst) # combined image
mask = cv2.bitwise_and(dst_mask_mouth,dst_mask_eyes)[:,:,0]

for channel in range(3):
    m = mask
    u = u_comb[:, :, channel]
    f = dst[:, :, channel]
    u1 = new_src[:, :, channel]

    beta_0 = 1   # TRY CHANGING
    beta = beta_0 * (1 - mask)

    # Aqui ja mho invento una mica

    vi, vj = poisson_editing.composite_gradients(u1, f, mask)
    b = poisson_editing.im_bwd_divergence(vi, vj)
    Au = poisson_editing.poisson_linear_operator(u, b)

    # Solve the linear system
    print(Au.shape, b.shape)
    u_final = spsolve(Au, b)
    # u_final = np.linalg.solve(Au, b)
    print(u_final)

    # Combine the final blended image with the original
    u_comb[:, :, channel] = u_final

cv2.imshow('Final result of Poisson blending', u_final)
# %%
