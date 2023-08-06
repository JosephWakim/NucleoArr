"""Model of arbitrary binding to a 1D polymer with sliding beads.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

import numpy as np
import sliding_nucleosome.nucleo_arr as nuc


def compute_matrix_product(T_all: np.ndarray):
    """Compute the matrix product of a set of transfer matrices.

    Notes
    -----
    When computing matrix products, factors are taken out of the product
    and stored in a list called alphas. The factors are taken out so that
    each time a new matrix multiplied into the product, the average value
    of the product remains zero. This ensures numerical stability.
    """
    alphas = []
    mat_prod = T_all[:, :, 0].copy()
    for i in range(1, T_all.shape[2]):
        mat_prod = np.matmul(mat_prod, T_all[:, :, i])
        norm_factor = np.average(mat_prod)
        alphas.append(norm_factor)
        mat_prod /= norm_factor
    return mat_prod, alphas


def compute_partition_function(T_all: np.ndarray):
    mat_prod, alphas = compute_matrix_product(T_all)
    return np.trace(mat_prod), alphas


def linker_move(nuc_arr: nuc.NucleosomeArray, ind: int):
    """Sample a new linker length.

    Notes
    -----
    This function is used to sample new linker lengths between nucleosomes.
    For a given linker index, the function computes the difference in free
    energy between the gamma = 1 state and the gamma = 0 state for that linker.
    The gamma = 1 and gamma = 0 states for the linker will be represented in
    the `T_all_1` and `T_all` arrays of `nuc_arr`, respectively. Once a new
    gamma is selected for the linker, both the T_all_1 and the `T_all` states
    will be updated to reflect the new gamma value.
    """
    if ind == nuc_arr.n_beads- 1:
        ind_p1 = 0
    else:
        ind_p1 = ind + 1
    # generate the two gamma values
    original_gamma = nuc_arr.gamma[ind]
    nuc_arr.T_all_1[:, :, ind] = \
        nuc_arr.get_transfer_matrix(ind, ind_p1, gamma_override=1)
    nuc_arr.T_all[:, :, ind] = \
        nuc_arr.get_transfer_matrix(ind, ind_p1, gamma_override=0)
    # compute the difference in free energy
    dF = get_dF(nuc_arr, ind)
    # Calculate the probability of gamma being one
    P_gt = get_P_gt(nuc_arr, dF)
    # Sample a new gamma value
    new_gamma = np.random.choice([0, 1], p=[1-P_gt, P_gt])
    if new_gamma == 1:
        nuc_arr.T_all[:, :, ind] = nuc_arr.T_all_1[:, :, ind]
    else:
        nuc_arr.T_all_1[:, :, ind] = nuc_arr.T_all[:, :, ind]
    nuc_arr.gamma[ind] = new_gamma
    # Sample a new linker length
    nuc_arr.linker_lengths[ind] = \
        sample_new_linker_length(nuc_arr, new_gamma)


def get_dF(nuc_arr: nuc.NucleosomeArray, ind: int) -> float:
    """Get the change in free energy associated with flipping the gamma value.
    """
    # Get free energies for gamma = 1 and gamma = 0 states
    trace_1, alphas_1 = compute_partition_function(nuc_arr.T_all_1)
    trace_0, alphas_0 = compute_partition_function(nuc_arr.T_all)
    # Compute the difference in free energy (don't forget to add the alphas)
    dF = -np.log(trace_1) + np.log(trace_0) - \
        np.sum(np.log(alphas_1)) + np.sum(np.log(alphas_0))
    return dF


def get_P_gt(nuc_arr: nuc.NucleosomeArray, dF: float) -> float:
    """Get the probability that the linker length is greater than cutoff `a`.
    """
    return nuc_arr.z / (np.exp(-dF) * (1 - nuc_arr.z) + nuc_arr.z)


def sample_new_linker_length(nuc_arr, new_gamma) -> float:
    """Sample a new linker length for the given linker index.
    """
    if new_gamma == 1:
        return nuc_arr.a + np.random.geometric(nuc_arr.p)
    else:
        return np.random.choice(nuc_arr.l_array, p=nuc_arr.p_lt)
