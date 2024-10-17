# %%
import cv2
import numpy as np
import poisson_editing
import time

# %%
# Configuration

image_name = "lena"
assert image_name in ["lena", "monalisa"]

# Manually adjust for blending strength
# The following values were tested to converge
if image_name == "lena":
    maxiter = 50
elif image_name == "monalisa":
    maxiter = 250
else:
    raise RuntimeError("Image name must be 'lena' or 'monalisa'")

input_folder = f"images/{image_name}"
output_folder = f"results/{image_name}"

# %%
# Load images

if image_name == "lena":
    src = cv2.imread(f"{input_folder}/girl.png")
    dst = cv2.imread(f"{input_folder}/lena.png")
elif image_name == "monalisa":
    src = cv2.imread(f"{input_folder}/ginevra.png")
    dst = cv2.imread(f"{input_folder}/lisa.png")
else:
    raise RuntimeError("Image name must be 'lena' or 'monalisa'")

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# %%

if image_name == "lena":
    # Load masks for eye swapping
    src_mask_eyes = cv2.imread(f"{input_folder}/mask_src_eyes.png", cv2.IMREAD_GRAYSCALE)
    dst_mask_eyes = cv2.imread(f"{input_folder}/mask_dst_eyes.png", cv2.IMREAD_GRAYSCALE)

    # Load masks for mouth swapping
    src_mask_mouth = cv2.imread(f"{input_folder}//mask_src_mouth.png", cv2.IMREAD_GRAYSCALE)
    dst_mask_mouth = cv2.imread(f"{input_folder}//mask_dst_mouth.png", cv2.IMREAD_GRAYSCALE)

    # Get the translation vectors
    t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes)
    t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth)

    # Cut out the relevant parts from the source image and shift them into the right position
    src_shifted_eyes = np.roll(src * (src_mask_eyes[:, :, None] > 0), shift=t_eyes, axis=(0, 1))
    src_shifted_mouth = np.roll(src * (src_mask_mouth[:, :, None] > 0), shift=t_mouth, axis=(0, 1))

    # Create a combined source image and mask
    src_shifted = src_shifted_eyes + src_shifted_mouth
    mask = ((src_shifted_eyes > 0) | (src_shifted_mouth > 0))[:,:,0].astype(np.uint8)
elif image_name == "monalisa":
    # No shift
    src_shifted = src
    # Load mask
    mask = (cv2.imread(f"{input_folder}/mask.png", cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
else:
    raise RuntimeError("Image name must be 'lena' or 'monalisa'")

# %%
# Save images
cv2.imwrite(f"{output_folder}/combined_source.png", src_shifted)
cv2.imwrite(f"{output_folder}/combined_mask.png", mask * 255)

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

    # Calculate beta for blending
    beta_0 = 1  # Note that since we're minimizing the problem, in our implementation beta_0 has no impact
    beta = beta_0 * (1 - m)
    h, w = beta.shape

    # Calculate the composite gradients
    vi, vj = poisson_editing.composite_gradients(u1, f, m)

    # Solve the linear system using conjugate gradient solver
    print(f"Solving the linear system for channel {channel}.")
    solve_start_time = time.time()

    # Compute the divergence
    b = np.where(beta > 0, f, 0) - poisson_editing.im_bwd_divergence(vi, vj)

    # Solve poisson
    u_final = poisson_editing.solve_poisson(u, beta, b.flatten(), maxiter=maxiter)

    # Reshape the result back to image shape
    u_comb[:, :, channel] = u_final.reshape((h,w))
    print(f"Solved linear system for channel {channel} in {time.time() - solve_start_time:.2f} seconds.")

    # Save images
    cv2.imwrite(f"{output_folder}/u1_channel_{channel}.png", u1 * 255)
    cv2.imwrite(f"{output_folder}/f_channel_{channel}.png", f * 255)
    cv2.imwrite(f"{output_folder}/composite_gradients_vi_channel_{channel}.png", vi * 255)
    cv2.imwrite(f"{output_folder}/composite_gradients_vj_channel_{channel}.png", vj * 255)
    cv2.imwrite(f"{output_folder}/im_bwd_divergence_b_channel_{channel}.png", b * 255)
    cv2.imwrite(f"{output_folder}/u_final_channel_{channel}.png", (u_final.reshape((h,w))) * 255)

# Display the final result
cv2.imwrite(f"{output_folder}/final_result_of_Poisson_blending_with_maxiter_{maxiter}.png", u_comb * 255)
# %%
