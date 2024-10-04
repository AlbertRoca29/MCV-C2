from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

@dataclass
class Parameters:
    hi: float
    hj: float

def laplace_equation(f, mask, param):

    ni = f.shape[0]
    nj = f.shape[1]

    # Add ghost boundaries on the image (for the boundary conditions)
    f_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ni_ext = f_ext.shape[0]
    nj_ext = f_ext.shape[1]
    f_ext[1: (ni_ext - 1), 1: (nj_ext - 1)] = f

    # Add ghost boundaries on the mask
    mask_ext = np.zeros((ni + 2, nj + 2), dtype=float)
    ndi_ext = mask_ext.shape[0]
    ndj_ext = mask_ext.shape[1]
    mask_ext[1 : ndi_ext - 1, 1 : ndj_ext - 1] = mask

    # Store memory for the A matrix and the b vector
    nPixels = (ni+2)*(nj+2) # Number of pixels

    # We will create A sparse, this is the number of nonzero positions
    # idx_Ai: Vector for the nonZero i index of matrix A
    # idx_Aj: Vector for the nonZero j index of matrix A
    # a_ij: Vector for the value at position ij of matrix A

    b = np.zeros(nPixels, dtype=float)

    # Vector counter
    idx=0
    idx_Ai=[]
    idx_Aj=[]
    a_ij=[]

    # North side boundary conditions
    i = 1
    for j in range(nj_ext):
        # from image matrix (i, j) coordinates to vectorial(p) coordinate
        p = j * (ni + 2) + i

        # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p)
        a_ij.insert(idx, 1)
        idx = idx + 1

        idx_Ai.insert(idx, p)
        idx_Aj.insert(idx, p + 1)
        a_ij.insert(idx, -1)
        idx = idx + 1

    # South side boundary conditions
    i = ni_ext
    for j in range(nj_ext):
        p = j * (ni + 2) + i
        # COMPLETE THE CODE

    # West side boundary conditions
    j = 0
    for i in range(ni_ext):
        p = j * (ni + 2) + i
        # COMPLETE THE CODE

    # East side boundary conditions
    j = nj_ext
    for i in range(ni_ext):
        p = (j - 1) * (ni + 2) + i
        # COMPLETE THE CODE

    # Looping over the pixels
    for j in range(1, nj + 1):
        for i in range(1, ni + 1):

            # from image matrix (i, j) coordinates to vectorial(p) coordinate
            p = j * (ni + 2) + i

            if mask_ext[i, j] == 1: # we have to in-paint this pixel

                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                # COMPLETE THE CODE

            else: # we do not have to in-paint this pixel

                # Fill Idx_Ai, idx_Aj and a_ij with the corresponding values and vector b
                # COMPLETE THE CODE

    idx_Ai_c = [i - 1 for i in idx_Ai]
    idx_Aj_c = [i - 1 for i in idx_Aj]

    # COMPLETE THE CODE (fill out the interrogation marks ???)
    # A = sparse(idx_Ai_c, idx_Aj_c, a_ij, ???, ???)
    # x = spsolve(A, b)

    u_ext = np.reshape(x,(ni+2, nj+2), order='F')
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