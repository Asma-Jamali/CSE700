import numpy as np

def gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the Gaussian kernel between two sets of vectors.
    
    Parameters:
    a (np.ndarray): 2D array of shape (d, na) representing na vectors of dimension d.
    b (np.ndarray): 2D array of shape (d, nb) representing nb vectors of dimension d.
    sigma (float): The kernel width parameter.
    
    Returns:
    np.ndarray: The computed Gaussian kernel matrix of shape (na, nb).
    """
    na = a.shape[1]
    nb = b.shape[1]
    k = np.zeros((na, nb), dtype=np.float64)
    
    inv_sigma = -0.5 / (sigma * sigma)
    
    for i in range(nb):
        for j in range(na):
            temp = a[:, j] - b[:, i]
            k[j, i] = np.exp(inv_sigma * np.dot(temp, temp))
    
    return k


def laplacian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the Laplacian kernel between two sets of vectors.
    
    Parameters:
    a (np.ndarray): 2D array of shape (d, na) representing na vectors of dimension d.
    b (np.ndarray): 2D array of shape (d, nb) representing nb vectors of dimension d.
    sigma (float): The kernel width parameter.
    
    Returns:
    np.ndarray: The computed Laplacian kernel matrix of shape (na, nb).
    """
    na = a.shape[1]
    nb = b.shape[1]
    k = np.zeros((na, nb), dtype=np.float64)
    
    inv_sigma = -1.0 / sigma
    
    for i in range(nb):
        for j in range(na):
            k[j, i] = np.exp(inv_sigma * np.sum(np.abs(a[:, j] - b[:, i])))
    
    return k



'''An implementation of local gaussian kernel for molecular similarity computation.'''
# purp. Compute Gaussian kernels between local atomic environments.
def local_gaussian_kernel(q1:np.array, q2:np.array, n1:np.array, n2:np.array, sigmas:np.array) -> np.array:
    """
    Computation of gaussian kernels between local atomic environments.
    
    Parameters:
    -----------
    q1 : ndarray (n_features, n_atoms1)
        Atomic descriptors for molecule set 1
    q2 : ndarray (n_features, n_atoms2)
        Atomic descriptors for molecule set 2
    n1 : ndarray (nm1,)
        Number of atoms for each molecule in set 1
    n2 : ndarray (nm2,)
        Number of atoms for each molecule in set 2
    sigmas : ndarray (nsigmas,)
        Gaussian kernel width parameters
        
    Returns:
    --------
    kernels : ndarray (nsigmas, nm1, nm2)
        Computed Gaussian kernels
    """
    # Get number of molecules and sigmas
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(sigmas)
    
    # Precompute -1.0 / (2 * sigma^2)
    inv_sigma2 = -0.5 / (sigmas ** 2)
    
    # Compute start indices for each molecule
    i_starts = np.cumsum(n1) - n1
    j_starts = np.cumsum(n2) - n2
    
    # Initialize kernel array
    kernels = np.zeros((nsigmas, nm1, nm2))
    
    # Compute kernel
    compute_local_gaussian_kernel(q1, q2, n1, n2, inv_sigma2, i_starts, j_starts, kernels)
    
    return kernels

# @njit(parallel=True)
def compute_local_gaussian_kernel(q1, q2, n1, n2, inv_sigma2, i_starts, j_starts, kernels):
    '''
    Compute Gaussian kernels for all sigmas.
    '''
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(inv_sigma2)
    
    max_n1 = np.max(n1)
    max_n2 = np.max(n2)
    
    for a in range(nm1):
        for b in range(nm2):
            ni = n1[a]
            nj = n2[b]
            
            # Compute pairwise distances
            atomic_distance = np.zeros((max_n1, max_n2))
            for i in range(ni):
                for j in range(nj):
                    diff = q1[:, i + i_starts[a]] - q2[:, j + j_starts[b]]
                    atomic_distance[i, j] = np.sum(diff ** 2)
            
            # Compute Gaussian kernels for all sigmas
            for k in range(nsigmas):
                total = 0.0
                for i in range(ni):
                    for j in range(nj):
                        total += np.exp(atomic_distance[i, j] * inv_sigma2[k])
                kernels[k, a, b] = total


def local_laplacian_kernel(q1: np.ndarray, q2: np.ndarray, n1: np.ndarray, n2: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    Compute local Laplacian kernels between two sets of molecular structures.
    """
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(sigmas)
    
    inv_sigma = -1.0 / sigmas
    kernels = np.zeros((nsigmas, nm1, nm2), dtype=np.float64)
    
    i_starts = np.cumsum(n1) - n1
    j_starts = np.cumsum(n2) - n2
    
    compute_local_laplacian_kernels(q1, q2, n1, n2, inv_sigma, i_starts, j_starts, kernels)
    
    return kernels


def compute_local_laplacian_kernels(q1, q2, n1, n2, inv_sigma, i_starts, j_starts, kernels):
    """
    Compute Laplacian kernels for all sigmas.
    """
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(inv_sigma)
    
    max_n1 = np.max(n1)
    max_n2 = np.max(n2)
    
    for a in range(nm1):
        for b in range(nm2):
            ni, nj = n1[a], n2[b]
            atomic_distance = np.zeros((max_n1, max_n2), dtype=np.float64)
            
            for i in range(ni):
                for j in range(nj):
                    diff = q1[:, i + i_starts[a]] - q2[:, j + j_starts[b]]
                    atomic_distance[i, j] = np.sum(np.abs(diff))
            
            for k in range(nsigmas):
                kernels[k, a, b] = np.sum(np.exp(atomic_distance[:ni, :nj] * inv_sigma[k]))