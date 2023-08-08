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
