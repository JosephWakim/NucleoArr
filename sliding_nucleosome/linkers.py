"""Model of arbitrary binding to a 1D polymer with sliding beads.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

import numpy as np
import sliding_nucleosomes.nucleo_arr as nuc


def compute_matrix_product(nuc_arr: nuc.NucleosomeArray):
    alphas = []
    mat_prod = nuc_arr.T_all[0].copy()
    for T in nuc_arr.T_all[1:]:
        mat_prod = np.matmul(mat_prod, T)
        norm_factor = np.average(mat_prod)
        alphas.append(norm_factor)
        mat_prod /= norm_factor
    return mat_prod, alphas


def compute_partition_function(nuc_arr: nuc.NucleosomeArray):
    mat_prod, alphas = compute_matrix_product(nuc_arr.T_all)
    return np.trace(mat_prod), alphas


def linker_move(nuc_arr: nuc.NucleosomeArray, ind: int):
    """Sample a new linker length
    """
    # Flip the gamma value in T_all_temp
    if nuc_arr.gamma[ind] == 0:
        T_test = nuc_arr.get_transfer_matrix(ind, gamma_override=1)
        original_gamma = 0
    else:
        T_test = nuc_arr.get_transfer_matrix(ind, gamma_override=0)
        original_gamma = 1
    nuc_arr.T_all_temp[ind] = T_test.copy()

    # Compute the difference in free energy associated with the gamma states
    dF = get_dF(nuc_arr)

    # Calculate the probability of gamma being one
    P_gt = get_P_gt(dF)

    # Sample a new gamma value
    nuc_arr.gamma[ind] = np.random.choice([0, 1], p=[1-P_gt, P_gt])

    # Update the transfer matrix
    if gamma[ind] == original_gamma:
        nuc_arr.T_all_temp[ind] = nuc_arr.T_all[ind].copy()

        # Sample new linker length
        pass

    else:
        nuc_arr.T_all[ind] = nuc_arr.T_all_temp[ind].copy()

        # Sample new linker length
        pass


def get_dF(nuc_arr: nuc.NucleosomeArray, ind: int) -> float:
    pass


def get_P_gt(dF) -> float:
    pass


def sample_new_linker_length() -> float:
    pass
