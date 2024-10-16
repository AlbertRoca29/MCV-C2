# %%
import cv2
import numpy as np
import poisson_editing
import time
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

# Save the images
cv2.imwrite('results/0-source_image.png', src)
cv2.imwrite('results/1-destination_image.png', dst)

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_GRAYSCALE)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('results/2-eyes_source_mask.png', src_mask_eyes)
cv2.imwrite('results/3-eyes_destination_mask.png', dst_mask_eyes)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_GRAYSCALE)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('results/4-mouth_source_mask.png', src_mask_mouth)
cv2.imwrite('results/5-mouth_destination_mask.png', dst_mask_mouth)

# %%
# Get the translation vectors 
t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes)
t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth)

# Cut out the relevant parts from the source image and shift them into the right position
src_shifted_eyes = np.roll(src * (src_mask_eyes[:, :, None] > 0), shift=t_eyes, axis=(0, 1))
src_shifted_mouth = np.roll(src * (src_mask_mouth[:, :, None] > 0), shift=t_mouth, axis=(0, 1))
cv2.imwrite('results/6-shifted_eyes_source.png', src_shifted_eyes)
cv2.imwrite('results/7-shifted_mouth_source.png', src_shifted_mouth)
# %%
# Create a combined source image and mask
src_shifted = src_shifted_eyes + src_shifted_mouth
mask = ((src_shifted_eyes > 0) | (src_shifted_mouth > 0))[:,:,0].astype(np.uint8)
cv2.imwrite('results/8-combined_source.png', src_shifted)
cv2.imwrite('results/9-combined_mask.png', mask * 255)

# Combined image
u_comb = np.copy(dst)
u_comb[mask>0] = src_shifted[mask>0]

cv2.imwrite('results/10-basic_approach.png', u_comb)

# %%

u_comb = np.zeros_like(dst).astype('float64')

# Iterate over each channel (R, G, B)
for channel in range(3):
    print(f"Processing channel: {channel}")

    # Get the mask, source, and destination channels
    m = mask
    u = u_comb.copy()
    u = u[:, :, channel]/255
    f = dst[:, :, channel]/255
    u1 = src_shifted[:, :, channel]/255
    cv2.imwrite(f"results/11-u1_{channel}.png", u1*255)
    cv2.imwrite(f"results/11-f_{channel}.png", f*255)

    # Calculate beta for blending
    beta_0 = 1   # Adjust for blending strength
    beta = beta_0 * (1 - m)
    
    h, w = beta.shape

    cv2.imwrite(f"results/12-beta.png", beta * 255)

    # Calculate the composite gradients
    vi, vj = poisson_editing.composite_gradients(u1, f, m)
    cv2.imwrite(f"results/13-composite_gradients_vi_{channel}.png", vi*255)
    cv2.imwrite(f"results/13-composite_gradients_vj_{channel}.png", vj*255)

    # Solve the linear system using conjugate gradient solver
    print(f"Solving the linear system for channel {channel}.")
    solve_start_time = time.time()

    # Compute the divergence
    b = np.where(beta > 0, f, 0) - poisson_editing.im_bwd_divergence(vi, vj)
    cv2.imwrite(f"results/14-im_bwd_divergence_b_{channel}.png", b*255)

    u_i = poisson_editing.solve_poisson(b.flatten(), beta, u.flatten())

    u_final = u_i.reshape((h,w))

    print(f"Solved linear system for channel {channel} in {time.time() - solve_start_time:.2f} seconds.")

    # Reshape the result back to image shape
    u_comb[:, :, channel] = u_final[:,:]
    
    cv2.imwrite(f"results/15-cg_u_final_{channel}.png", u_final*255)

# Display the final result
cv2.imwrite('results/XX-final_result_of_Poisson_blending.png', u_comb*255)
# %%