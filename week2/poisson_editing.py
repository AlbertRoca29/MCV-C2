import numpy as np
from scipy.signal import correlate2d
from scipy.sparse.linalg import LinearOperator, cg


def im_fwd_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward gradient of an image.

    The gradient is computed by shifting the image one pixel and subtracting it
    from the original. It calculates both vertical and horizontal gradients.

    Parameters:
    - image: 2D numpy array representing the input image.

    Returns:
    - grad_i: Vertical gradient (shift along axis 0).
    - grad_j: Horizontal gradient (shift along axis 1).
    """
    grad_i = np.roll(image, -1, axis=0) - image  # Vertical gradient -> (H x W)
    grad_j = np.roll(image, -1, axis=1) - image  # Horizontal gradient -> (H x W)
    return grad_i, grad_j


def im_bwd_divergence(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Computes the backward divergence of two input gradients (inverse of the gradient).

    Parameters:
    - im1: Vertical component of the gradient.
    - im2: Horizontal component of the gradient.

    Returns:
    - div: 2D numpy array representing the divergence.
    """
    div_i = im1 - np.roll(im1, 1, axis=0)  # Inverse vertical gradient -> (H x W)
    div_j = im2 - np.roll(im2, 1, axis=1)  # Inverse horizontal gradient -> (H x W)
    return div_i + div_j


def composite_gradients(u1: np.ndarray, u2: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a composite vector field by combining the forward gradients of u1 and u2.

    Where the mask is 1, the composite gradient follows u1; where the mask is 0,
    it follows u2.

    Parameters:
    - u1: First input 2D array (image or field).
    - u2: Second input 2D array (image or field).
    - mask: Binary mask that defines which gradient to follow at each pixel.

    Returns:
    - vi: Vertical component of the composite gradient.
    - vj: Horizontal component of the composite gradient.
    """
    grad_u1_i, grad_u1_j = im_fwd_gradient(u1)
    grad_u2_i, grad_u2_j = im_fwd_gradient(u2)

    vi = mask * grad_u1_i + (1 - mask) * grad_u2_i  # Vertical composite gradient -> (H x W)
    vj = mask * grad_u1_j + (1 - mask) * grad_u2_j  # Horizontal composite gradient -> (H x W)
    return vi, vj


def poisson_linear_operator(u: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Implements the Poisson linear operator for the Poisson editing problem.

    This operator applies the matrix A, which includes a combination of the image
    values and their divergence.

    Parameters:
    - u: 2D numpy array representing the current estimate of the solution.
    - beta: Binary mask or constraint matrix.

    Returns:
    - Au: Result of applying the linear operator to u.
    """
    grad_u_i, grad_u_j = im_fwd_gradient(u)
    div_u = im_bwd_divergence(grad_u_i, grad_u_j)

    Au = np.where(beta > 0, u, 0) - div_u  # Apply the operator, conditional on beta
    return Au


def get_translation(original_img: np.ndarray, translated_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates the translation vector (shift) between two images based on cross-correlation.

    Parameters:
    - original_img: The original 2D image (numpy array).
    - translated_img: The translated version of the original image.

    Returns:
    - shift_y: Estimated vertical translation (in pixels).
    - shift_x: Estimated horizontal translation (in pixels).
    """
    correlation = correlate2d(translated_img, original_img, boundary='fill', mode='same')

    # Calculate the weighted mean of the correlation to estimate the shift
    h, w = correlation.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    intensities = correlation.flatten()

    mean_x = np.sum(x_coords.flatten() * intensities) / np.sum(intensities)
    mean_y = np.sum(y_coords.flatten() * intensities) / np.sum(intensities)

    shift_y, shift_x = round(mean_y), round(mean_x)

    # Adjust shifts to center
    shift_y -= correlation.shape[0] // 2
    shift_x -= correlation.shape[1] // 2

    return shift_y, shift_x


def solve_poisson(u_init: np.ndarray, beta: np.ndarray, b: np.ndarray, tol=1e-5, maxiter=50) -> np.ndarray:
    """
    Solves the Poisson equation using the Conjugate Gradient (CG) method.

    Parameters:
    - u_init: Initial guess for the solution (2D numpy array).
    - beta: Binary mask or constraint matrix (2D numpy array).
    - b: Right-hand side vector (2D numpy array).
    - tol: Tolerance for the solver convergence (default is 1e-5).
    - maxiter: Maximum number of iterations for CG (default is 1000).

    Returns:
    - u: Solution to the linear system (flattened 1D numpy array).
    """

    # Define a lambda function representing the action of the matrix A for the CG method.
    # This operator reshapes u (1D array) back to 2D and applies poisson_linear_operator,
    # then flattens the result back into 1D form for compatibility with CG.
    A_operator = lambda u: poisson_linear_operator(u.reshape(beta.shape), beta).flatten()

    # Create a linear operator to represent the matrix A in the CG method.
    # It specifies how to multiply vectors with A (using A_operator), the data type,
    # and the size of the matrix (b.size is the number of elements in the flattened image).
    A_linear_operator = LinearOperator(matvec=A_operator, dtype=b.dtype, shape=(b.size, b.size))

    # Solve the linear system A*u = b using the Conjugate Gradient (CG) method.
    # The initial guess for the solution is u_init, flattened to a 1D array.
    # 'rtol' is the relative tolerance for convergence, and 'maxiter' is the maximum number of iterations.
    u, info = cg(A_linear_operator, b.flatten(), x0=u_init.flatten(), rtol=tol, maxiter=maxiter)

    # Check if the CG solver converged. If 'info' is not 0, print a warning message with the error code.
    if info != 0:
        print(f"Conjugate Gradient did not converge. Info: {info}")
    else:
        print(f"Conjugate Gradient has converged!. Info: {info}")

    return u
