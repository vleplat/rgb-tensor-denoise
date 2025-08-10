# Module: decomposition.py
import copy
import tensorly as tl
from .tucker2_lib import tucker_denoising, tucker_truncatedhosvd_init
from .tt_lib import tt_nestedtk2, tt_adcu
import numpy as np

def decompose_and_denoise_patch(X, method="tucker", sigma=0.075, tt_opts=None, tt_rank=None, reshape_for_tt=False, target_order=10):
    """
    Unified interface to denoise patches using Tucker or TT.

    Parameters:
        X : ndarray
            Patch tensor of shape (H, W, C, K)
        method : str
            'tucker', 'tt_fixed', or 'tt_adaptive'
        sigma : float
            Noise std.
        tt_opts : dict
            Options for TT adaptive mode.
        tt_rank : int or list
            Ranks for TT fixed mode.
        reshape_for_tt : bool
            Whether to reshape before TT.
        target_order : int
            Desired order of reshaped tensor

    Returns:
        X_hat : ndarray
            Denoised patch
        model : Tensor factors (core + U for Tucker, TT cores for TT)
        reshape_for_tt : bool
            Whether reshaping was applied
    """
    if method == "tucker":
        X_hat, G, U = tucker_decompose_and_denoise(X, sigma)
        return X_hat, (G, U), reshape_for_tt
    elif method == "tt_fixed":
        tt_tensor = tt_decompose_fixed_ranks(X, tt_rank, reshape_for_tt, target_order=target_order)
        X_hat = tl.tt_to_tensor(tt_tensor)
        if reshape_for_tt:
            X_hat = reshape_to_original(X_hat, X.shape)
        return X_hat, tt_tensor, reshape_for_tt
    elif method == "tt_adaptive":
        tt_tensor, _ = tt_decompose_adaptive(X, sigma, tt_opts, reshape_for_tt, target_order=target_order)
        X_hat = tl.tt_to_tensor(tt_tensor)
        if reshape_for_tt:
            X_hat = reshape_to_original(X_hat, X.shape)
        return X_hat, tt_tensor, reshape_for_tt
    else:
        raise ValueError(f"Unknown method: {method}")



def reshape_to_higher_order(patch, target_order=6):
    """
    Reshape a 4D tensor (H, W, C, K) to a higher-order tensor.
    This function reshapes using balanced block sizes (powers of 2 or 3 preferred).
    """
    import math

    original_shape = patch.shape
    total_size = np.prod(original_shape)

    # Simple greedy decomposition: build shape with small primes
    def greedy_factors(n, target_order):
        factors = []
        primes = [2, 3, 5, 7]
        for p in primes:
            while len(factors) < target_order - 1 and n % p == 0:
                factors.append(p)
                n //= p
        factors.append(n)
        return factors

    new_shape = greedy_factors(total_size, target_order)

    if np.prod(new_shape) != total_size:
        raise ValueError(f"Cannot factor {total_size} cleanly into {target_order} parts. Try different shape or order.")

    reshaped = patch.reshape(new_shape)
    return reshaped, original_shape


def reshape_to_original(patch_reshaped, original_shape):
    """Reverse reshape back to original shape."""
    return patch_reshaped.reshape(original_shape)

def tucker_decompose_and_denoise(X, sigma, max_iter=3, tol=1e-5):
    """Tucker decomposition + denoising with HOSVD init.

    Returns:
        - X_hat: reconstructed tensor
        - G: core tensor
        - U: list of factor matrices
    """
    decomposemodes = np.arange(X.ndim)
    approx_bound = sigma**2 * X.size
    Utr, Rtr, err, _ = tucker_truncatedhosvd_init(X, approx_bound, decomposemodes)
    Gtr = tl.tenalg.multi_mode_dot(X, Utr, modes=decomposemodes, transpose=True)
    Xtl = tl.tenalg.multi_mode_dot(Gtr, Utr, modes=decomposemodes)

    X_mean = np.mean(X)
    U, G, aprxerror, noparams, rankR = tucker_denoising(
        X - X_mean, copy.deepcopy(Utr), max_iter, tol, sigma, decomposemodes,
        exacterrorbound=None, precision=None, verbose=False
    )
    X_hat = tl.tenalg.multi_mode_dot(G, U, modes=decomposemodes) + X_mean
    return X_hat, G, U

def tt_decompose_fixed_ranks(X, rank, reshape_for_tt=False, spatial_splits=(2, 2), target_order=10):
    """
    TT decomposition using TensorLy with fixed ranks.

    Parameters:
        X : ndarray
            Patch tensor of shape (H, W, C, K).
        rank : int or list
            Desired TT-ranks.
        reshape_for_tt : bool
            Whether to reshape the input to a higher-order tensor.
        spatial_splits : tuple
            Factorization for spatial dims.

    Returns:
        tt_tensor : TTTensor
    """
    if reshape_for_tt:
        X, _ = reshape_to_higher_order(X, target_order=target_order)
    return tl.decomposition.tensor_train(X, rank)

def tt_decompose_adaptive(X, sigma, opts=None, reshape_for_tt=False, spatial_splits=(2, 2), verbose=False, target_order=10):
    """
    TT decomposition using nested Tucker + ADCU for error-bounded approximation.

    Parameters:
        X : ndarray
            Input tensor (e.g. d x d x 3 x K).
        sigma : float
            Noise standard deviation.
        opts : dict
            ADCU options.
        reshape_for_tt : bool
            Whether to reshape to higher-order.
        spatial_splits : tuple
            Splitting pattern for reshaping spatial dims.
        verbose : bool
            Print approximation info.

    Returns:
        Xt : TTTensor
        err : list of errors per iteration
    """
    if reshape_for_tt:
        X, _ = reshape_to_higher_order(X, target_order=target_order)


    approx_bound = sigma**2 * X.size
    Yttnested, approx_error_init = tt_nestedtk2(X, approx_bound * 1.05)

    if verbose:
        print(f"[init] Nested Tucker2 error: {approx_error_init:.4e}")

    Xt = copy.deepcopy(Yttnested)
    rankX = np.array(Xt.rank)

    if opts is None:
        opts = {
            "init": "exac",
            "compression": False,
            "noise_level": sigma,
            "exacterrorbound": True,
            "tol": 1e-6,
            "maxiters": 50,
        }

    Xt, err = tt_adcu(X, Xt, rankX, opts)
    return Xt, err

def tt_num_parameters(tt_tensor):
    """
    Compute total number of parameters in a TT decomposition.
    
    Parameters:
        tt_tensor : TTTensor

    Returns:
        int
            Number of scalar parameters in all cores
    """
    return sum(np.prod(core.shape) for core in tt_tensor.factors)

def tt_compression_ratio(tt_tensor, original_shape):
    """
    Compute compression ratio of TT w.r.t original tensor.

    Parameters:
        tt_tensor : TTTensor
        original_shape : tuple

    Returns:
        float
            Ratio: (#TT parameters) / (#original parameters)
    """
    total_params = tt_num_parameters(tt_tensor)
    original_params = np.prod(original_shape)
    return total_params / original_params