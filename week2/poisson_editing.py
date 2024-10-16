import numpy as np
from scipy.signal import correlate2d
from scipy.sparse.linalg import LinearOperator, cg

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
    
    grad_u_i, grad_u_j = im_fwd_gradient(u)
    
    div_u = im_bwd_divergence(grad_u_i, grad_u_j)
    
    Au = np.where(beta > 0, u, 0) - div_u
    
    return Au


def get_translation(original_img: np.ndarray, translated_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    correlation = correlate2d(translated_img, original_img, boundary='fill', mode='same')
    
    # The correlation is having some issues, but is well centered (got it at plotting)
    # So I am getting the value at center doing the weightened mean

    h, w = correlation.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    intensities = correlation.flatten()
    mean_x = np.sum(x_coords.flatten() * intensities) / np.sum(intensities)
    mean_y = np.sum(y_coords.flatten() * intensities) / np.sum(intensities)

    shift_y,shift_x = round(mean_y),round(mean_x)

    shift_y -= correlation.shape[0] // 2
    shift_x -= correlation.shape[1] // 2
    
    return shift_y, shift_x

def solve_poisson(b: np.ndarray, beta: np.ndarray, u_init: np.ndarray, tol=1e-5, maxiter=1000):
    """
    Solves the linear system Au = b using the Conjugate Gradient method.
    
    Parameters:
    - b: Right-hand side vector (numpy array).
    - beta: Binary mask or constraint matrix (numpy array).
    - u_init: Initial guess for the solution (numpy array).
    - tol: Tolerance for the solver convergence.
    - maxiter: Maximum number of iterations.
    
    Returns:
    - u: Solution vector (numpy array).
    """
    def A_operator(u):
        h, w = beta.shape
        return poisson_linear_operator(u.reshape((h,w)), beta)

    A_linear_operator = LinearOperator(matvec=A_operator, dtype=b.dtype, shape=(b.size, b.size))
    
    u, info = cg(A_linear_operator, b, x0=u_init, rtol=tol, maxiter=maxiter)
    
    if info != 0:
        print(f"Conjugate Gradient did not converge. Info: {info}")
    
    return u