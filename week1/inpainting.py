from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

@dataclass
class Parameters:
    hi: float
    hj: float

def laplace_equation(f: np.ndarray, mask: np.ndarray, param) -> np.ndarray:
    ni = f.shape[0]
    nj = f.shape[1]

    # Add ghost boundaries on the image (for the boundary conditions)
    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0] # = ni+2
    nj_ext = f_ext.shape[1]
    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f

    # Add ghost boundaries on the mask
    mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    mask_ext[1: ni_ext - 1, 1: nj_ext - 1] = mask

    # Store memory for the A matrix and the b vector
    nPixels = ni_ext*nj_ext # Number of pixels

    # We will create A sparse, this is the number of nonzero positions

    # idx_Ai: Vector for the nonZero i index of matrix A
    # idx_Aj: Vector for the nonZero j index of matrix A
    # a_ij: Vector for the value at position ij of matrix A

    b = np.zeros(nPixels, dtype=float)

    # Vector counter
    idx_Ai, idx_Aj, a_ij = [], [], []

    # North side boundary conditions
    i = 0
    for j in range(nj_ext):
        p = j * ni_ext + i
        idx_Ai.extend([p, p])
        idx_Aj.extend([p, p + 1])
        a_ij.extend([1, -1])
        b[p] = 0

    # South side boundary conditions
    i = ni_ext - 1
    for j in range(nj_ext):
        p = j * ni_ext + i
        idx_Ai.extend([p, p])
        idx_Aj.extend([p, p - 1])
        a_ij.extend([1, -1])
        b[p] = 0

    # West side boundary conditions
    j = 0
    for i in range(ni_ext):
        p = j * ni_ext + i
        idx_Ai.extend([p, p])
        idx_Aj.extend([p, p + ni_ext])
        a_ij.extend([1, -1])
        b[p] = 0

    # East side boundary conditions
    j = nj_ext - 1
    for i in range(ni_ext):
        p = j * ni_ext + i
        idx_Ai.extend([p, p - ni_ext])
        idx_Aj.extend([p, p])
        a_ij.extend([1, -1])
        b[p] = 0

    # Looping over the pixels
    for j in range(nj_ext):
        for i in range(ni_ext):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = j * ni_ext + i

            if mask_ext[i, j] > 0 : # we have to in-paint this pixel
                idx_Ai.extend([p, p         , p         , p    , p])
                idx_Aj.extend([p, p - ni_ext, p + ni_ext, p - 1, p + 1])
                a_ij.extend([4, -1, -1, -1, -1])  # Coefficients
                b[p] = 0  # Right-hand side for inpainting (can be adjusted)

            else: # we do not have to in-paint this pixel
                idx_Ai.append(p)
                idx_Aj.append(p)  
                a_ij.append(1) 
                b[p] = f_ext[i, j] 

    A = sparse(idx_Ai, idx_Aj, a_ij, nPixels, nPixels)
    x = spsolve(A, b)

    u_ext = np.reshape(x,(ni_ext, nj_ext), order='F')
    u_ext_i = u_ext.shape[0]
    u_ext_j = u_ext.shape[1]

    u = np.full((ni, nj), u_ext[1:u_ext_i-1, 1:u_ext_j-1], order='F')
    return u

def sparse(i, j, v, m, n):
    """
    Create and compress a matrix that have many zeros
    Parameters:
        i: 1-D array representing the index 1 values
            Size n1
        j: 1-D array representing the index 2 values
            Size n1
        v: 1-D array representing the values
            Size n1
        m: integer representing x size of the matrix >= n1
        n: integer representing y size of the matrix >= n1
    Returns:
        s: 2-D array
            Matrix full of zeros excepting values v at indexes i, j
    """
    return csr_matrix((v, (i, j)), shape=(m, n))
