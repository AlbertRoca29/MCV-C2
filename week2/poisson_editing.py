import numpy as np
from scipy.signal import correlate2d

# Forward Gradient
def im_fwd_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
    grad_i = np.roll(image, -1, axis=0) - image # Vertical
    grad_j = np.roll(image, -1, axis=1) - image # Horizontal
    return grad_i, grad_j

# Backward Divergence (inverse of the gradient)
def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    div_i = im1 - np.roll(im1, 1, axis=0)  # Inverse vertical
    div_j = im2 - np.roll(im2, 1, axis=1)  # Inverse horizontal
    return div_i + div_j

def composite_gradients(u1: np.ndarray, u2: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a vector field v by combining the forward gradient of u1 and u2.
    For pixels where the mask is 1, the composite gradient v must coincide
    with the gradient of u1. When mask is 0, the composite gradient v must coincide
    with the gradient of u2.

    :return vi: composition of i components of gradients (vertical component)
    :return vj: composition of j components of gradients (horizontal component)
    """

    grad_u1_i, grad_u1_j = im_fwd_gradient(u1)
    grad_u2_i, grad_u2_j = im_fwd_gradient(u2)

    vi = mask * grad_u1_i + (1 - mask) * grad_u2_i  # Vertical
    vj = mask * grad_u1_j + (1 - mask) * grad_u2_j  # Horizontal
    return vi, vj

def poisson_linear_operator(u: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Implements the action of the matrix A in the quadratic energy associated
    to the Poisson editing problem.
    """
    laplace_u = im_bwd_divergence(*im_fwd_gradient(u))
    Au = laplace_u * beta
    return Au


def get_translation(original_img: np.ndarray, translated_img: np.ndarray, *part: str) -> tuple[np.ndarray, np.ndarray]:
    
    # Compute cross-correlation between the original and translated images
    correlation = correlate2d(translated_img[:,:,0], original_img[:,:,0], boundary='symm', mode='same')
    
    # Find the location of the maximum correlation, which corresponds to the translation
    max_corr_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Calculate the shift vector based on the maximum correlation index
    shift_y = max_corr_idx[0] - original_img.shape[0] // 2
    shift_x = max_corr_idx[1] - original_img.shape[1] // 2
    
    return shift_y, shift_x

# def get_translation(original_img: np.ndarray, translated_img: np.ndarray, *part: str):

    # For the eyes mask:
    # The top left pixel of the source mask is located at (x=115, y=101)
    # The top left pixel of the destination mask is located at (x=123, y=125)
    # This gives a translation vector of (dx=8, dy=24)

    # For the mouth mask:
    # The top left pixel of the source mask is located at (x=125, y=140)
    # The top left pixel of the destination mask is located at (x=132, y=173)
    # This gives a translation vector of (dx=7, dy=33)

    # The following shifts are hard coded:
    if part[0] == "eyes":
        return (24, 8)
    elif part[0] == "mouth":
        return (33, 7)
    else:
        return (0, 0)

    # Here on could determine the shift vector programmatically,
    # given an original image/mask and its translated version.
    # Idea: using maximal cross-correlation (e.g., scipy.signal.correlate2d), or similar.