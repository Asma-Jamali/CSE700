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
