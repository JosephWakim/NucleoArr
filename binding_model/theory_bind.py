"""Generate the theoretical binding fraction for a nucleosme array.

By:        Joseph Wakim
Group:     Spakowitz Lab
Date:      07 Aug 2023
"""

from typing import Optional
import numpy as np
import sliding_nucleosome.nucleo_arr as nuc
import sliding_nucleosome.linkers as link


def differentiate_transfer_matrix(T):
    """Differentiate a transfer matrix with respect to mu_t.

    Notes
    -----
    This function assumes that the transfer matrix represents a nucleosome
    array with a single binder. The function will be updated in the future
    to work with multiple binders. When multiple binders are involved, we
    will need to be more mindful when differentiating the transfer matrices.

    Parameters
    ----------
    T : np.ndarray
        Transfer matrix to be differentiated

    Returns
    -------
    np.ndarray
        Derivative of the transfer matrix with respect to mu_t
    """
    dT = T.copy()
    dT[0, 0] = 0
    dT[0, 1] *= 0.5
    dT[1, 0] *= 0.5
    dT[1, 2] *= 1.5
    dT[2, 1] *= 1.5
    dT[2, 2] *= 2
    return dT


def compute_theoretical_binding_fraction(
    nuc_arr: nuc.NucleosomeArray,
    binder_ind: Optional[int] = 0
) -> float:
    """Compute the theoretical binding fraction along a nucleosome array.

    Notes
    -----
    This current version of the function only works for nucleosome arrays
    with a single binder. The function will be updated in the future to
    work with multiple binders. When multiple binders are involved, we will
    need to be more mindful when differentiating the transfer matrices.

    Parameters
    ----------
    nuc_arr : nuc.NucleosomeArray
        Nucleosome array for which the theoretical binding fraction is to be
        computed
    binder_ind : Optional[int]
        Index of the binder for which binding fraction is to be calculated
        (default = 0)
    """
    # How many binder states exist?
    Nr = nuc_arr.Nbi[binder_ind] + 1
    # Compute derivatives
    dT_all = [
        differentiate_transfer_matrix(nuc_arr.T_all[:, :, i])
        for i in range(nuc_arr.n_beads)
    ]
    # Compute partition function and normalization factors
    Z, alphas = link.compute_partition_function(nuc_arr.T_all)
    # Pre-compute left and right partial matrix product
    N = nuc_arr.n_beads
    T_left = {}
    T_right = {}
    for i in range(N):
        if i == 0:
            T_left[i] = np.identity(Nr)
        else:
            T_left[i] = np.matmul(T_left[i-1], nuc_arr.T_all[:, :, i-1]) / alphas[i-1]
    for ind in range(N):
        i = N - ind - 1
        if i == N-1:
            T_right[i] = np.identity(Nr)
        else:
            T_right[i] = np.matmul(nuc_arr.T_all[:, :, i+1], T_right[i+1]) / alphas[i]
    # Compute derivative
    dZ = np.zeros((Nr, Nr))
    for i in range(N):
        dZ += np.matmul(np.matmul(T_left[i], dT_all[i]), T_right[i])
    # Calculate theoretical binding fraction
    theta_theory = 1 / (nuc_arr.Nbi[0] * N * Z) * np.trace(dZ)
    return theta_theory


def find_mu(
    nuc_arr: nuc.NucleosomeArray,
    lower_input: float,
    upper_input: float,
    setpoint: float,
    binder_ind: Optional[int] = 0,
    iter_: Optional[int] = 0,
    max_iters: Optional[int] = 100,
    rtol: Optional[float] = 0.05
):
    """Find chemical potential that produces desired binding fraction.

    Notes
    -----
    This function uses a binary search to find the chemical potential that
    produces the desired binding fraction. The function assumes that the
    binding fraction is a monotonically increasing function of the chemical
    potential. The function will be updated in the future to work with
    multiple binders.

    Parameters
    ----------
    nuc_arr : nuc.NucleosomeArray
        Nucleosome array for which the theoretical binding fraction is to be
        computed
    lower_input : float
        Lower bound of the chemical potential search
    upper_input : float
        Upper bound of the chemical potential search
    setpoint : float
        Desired binding fraction
    binder_ind : Optional[int]
        Index of the binder for which binding fraction is to be calculated
        (default = 0)
    iter_ : Optional[int]
        Iteration number (default = 0)
    max_iters : Optional[int]
        Maximum number of iterations (default = 100)
    rtol : Optional[float]
        Relative tolerance (default = 0.05)

    Returns
    -------
    float
        Chemical potential that produces the desired binding fraction
    """
    # Update iteration
    iter_ += 1
    if (iter_ % 10) == 0:
        print(f"Iteration: {iter_}")
    # Update the chemical potential and calculate the binding fraction
    test_mu = (lower_input + upper_input) / 2
    nuc_arr.mu[binder_ind] = test_mu
    nuc_arr.get_all_transfer_matrices()
    bind_frac_ = compute_theoretical_binding_fraction(nuc_arr)
    # Base Cases
    if iter_ >= max_iters:
        print("Maximum number of iterations have been met.")
        return test_mu
    elif np.abs((bind_frac_ - setpoint)) / setpoint <= rtol:
        print("Converged!")
        return test_mu
    # Recursive Cases
    else:
        # If binding fraction was too high, reiterate on lower half of mu
        if bind_frac_ > setpoint:
            next_mu = find_mu(
                nuc_arr, lower_input, test_mu, setpoint,
                iter_, max_iters, rtol
            )
        # If binding fraction was too low, reiterate on upper half of mu
        else:
            next_mu = find_mu(
                nuc_arr, test_mu, upper_input, setpoint,
                iter_, max_iters, rtol
            )
        return next_mu
