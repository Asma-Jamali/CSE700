"""This is a centralized library for all the kernels used by the kernel ridge regression."""

import numpy as np
import gpytorch
import torch
from gpytorch.kernels import Kernel

# FD. gaussian_kernel()
def gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the Gaussian kernel between two sets of vectors.
    
    Types and Parameters:
    a (np.ndarray): 2D array of shape (d, na) representing na vectors of dimension d.
    b (np.ndarray): 2D array of shape (d, nb) representing nb vectors of dimension d.
    sigma (float): The kernel width parameter.
    
    Returns:
    np.ndarray: The computed Gaussian kernel matrix of shape (na, nb).
    """
    na = a.shape[1]
    nb = b.shape[1]
    kernel = np.zeros((na, nb), dtype=np.float64)
    
    inv_sigma = -0.5 / (sigma * sigma)
    
    for i in range(nb):
        for j in range(na):
            temp = a[:, j] - b[:, i]
            kernel[j, i] = np.exp(inv_sigma * np.dot(temp, temp))
    
    return kernel

# FD. laplacian_kernel()
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
    kernel = np.zeros((na, nb), dtype=np.float64)
    
    inv_sigma = -1.0 / sigma
    
    for i in range(nb):
        for j in range(na):
            kernel[j, i] = np.exp(inv_sigma * np.sum(np.abs(a[:, j] - b[:, i])))
    
    return kernel

# FD. local_gaussian_kernel()
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


# FD. local_laplacian_kernel()
def local_laplacian_kernel(q1: np.ndarray, q2: np.ndarray, n1: np.ndarray, n2: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """
    Compute local Laplacian kernels between two sets of molecular structures.
    """
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
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(sigmas)
    
    inv_sigma = -1.0 / sigmas
    kernels = np.zeros((nsigmas, nm1, nm2), dtype=np.float64)
    
    i_starts = np.cumsum(n1) - n1
    j_starts = np.cumsum(n2) - n2
    
    compute_local_laplacian_kernels(q1, q2, n1, n2, inv_sigma, i_starts, j_starts, kernels)
    
    return kernels

# CD. TanimotoKernel(Kernel)
class TanimotoKernel(Kernel):
    r"""
     Computes a covariance matrix based on the Tanimoto kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

     .. math::

        \begin{equation*}
        k_{\text{Tanimoto}}(\mathbf{x}, \mathbf{x'}) = \frac{\langle\mathbf{x},
        \mathbf{x'}\rangle}{\left\lVert\mathbf{x}\right\rVert^2 + \left\lVert\mathbf{x'}\right\rVert^2 -
        \langle\mathbf{x}, \mathbf{x'}\rangle}
        \end{equation*}

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)

    def covar_dist(
        self,
        x1,
        x2,
        last_dim_is_batch=False,
        **params,
    ):
        r"""This is a helper method for computing the bit vector similarity between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        return self.batch_tanimoto_sim(x1, x2)
        
        
    def batch_tanimoto_sim(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Tanimoto similarity between two batched tensors, across last 2 dimensions.
        eps argument ensures numerical stability if all zero tensors are added. Tanimoto similarity is proportional to:

        :math:`(<x, y>) / (||x||^2 + ||y||^2 - <x, y>)`

        where x and y may be bit or count vectors or in set notation:

        :math:`|A \\cap B| / |A| + |B| - |A \\cap B|`

        Args:
            x1: `[b x n x d]` Tensor where b is the batch dimension
            x2: `[b x m x d]` Tensor
            eps: Float for numerical stability. Default value is 1e-6

        Returns:
            Tensor denoting the Tanimoto similarity.
        """

        if x1.ndim < 2 or x2.ndim < 2:
            raise ValueError("Tensors must have a batch dimension")

        dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
        x1_norm = torch.sum(x1**2, dim=-1, keepdims=True)
        x2_norm = torch.sum(x2**2, dim=-1, keepdims=True)

        tan_similarity = (dot_prod + eps) / (
            eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
        )

        return tan_similarity.clamp_min_(
            0
        )  # zero out negative values for numerical stability

    
# CD. DiceKernel(Kernel)
class DiceKernel(Kernel):
    r"""
     Computes a covariance matrix based on the Dice kernel
     between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

     .. math::

        \begin{equation*}
        k_{\text{Dice}}(\mathbf{x}, \mathbf{x'}) = \frac{2\langle\mathbf{x},
        \mathbf{x'}\rangle}{\left\lVert\mathbf{x}\right\rVert + \left\lVert\mathbf{x'}\right\rVert}
        \end{equation*}

    .. note::

     This kernel does not have an `outputscale` parameter. To add a scaling parameter,
     decorate this kernel with a :class:`gpytorch.test_kernels.ScaleKernel`.

     Example:
         >>> x = torch.randint(0, 2, (10, 5))
         >>> # Non-batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(DiceKernel())
         >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
         >>>
         >>> batch_x = torch.randint(0, 2, (2, 10, 5))
         >>> # Batch: Simple option
         >>> covar_module = gpytorch.kernels.ScaleKernel(DiceKernel())
         >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(DiceKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)

    def covar_dist(
        self,
        x1,
        x2,
        last_dim_is_batch=False,
        **params,
    ):
        r"""This is a helper method for computing the bit vector similarity between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        return self.batch_dice_sim(x1, x2)

    def batch_dice_sim(self,
        x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Dice similarity between two batched tensors, across last 2 dimensions.
        eps argument ensures numerical stability if all zero tensors are added.

        :math:`(2 * <x1, x2>) / (|x1| + |x2|)`

        Where || is the L1 norm and <.> is the inner product

        Args:
            x1: `[b x n x d]` Tensor where b is the batch dimension
            x2: `[b x m x d]` Tensor
            eps: Float for numerical stability. Default value is 1e-6
        Returns:
            Tensor denoting the Dice similarity.
        """

        if x1.ndim < 2 or x2.ndim < 2:
            raise ValueError("Tensors must have a batch dimension")

        # Compute L1 norm
        x1_norm = torch.sum(x1, dim=-1, keepdims=True)
        x2_norm = torch.sum(x2, dim=-1, keepdims=True)
        dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))

        dice_similarity = (2 * dot_prod + eps) / (
            x1_norm + torch.transpose(x2_norm, -1, -2) + eps
        )

        return dice_similarity.clamp_min_(
            0
        )  # zero out negative values for numerical stability
