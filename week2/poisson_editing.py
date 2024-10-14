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
    
    # Find the peak in the correlation matrix
    shift_y, shift_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Center the shift values around the middle of the image
    shift_y -= correlation.shape[0] // 2
    shift_x -= correlation.shape[1] // 2
    
    return shift_y, shift_x
