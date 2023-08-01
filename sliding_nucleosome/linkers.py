"""Model of arbitrary binding to a 1D polymer with sliding beads.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

import os
from math import comb
import numpy as np
import pandas as pd
import sliding_nucleosomes.nucleo_arr as nuc


def compute_partition_function(transfer_matrices):
    mat_prod, alphas = compute_matrix_product(transfer_matrices)
    return np.trace(mat_prod), alphas


def compute_matrix_product(transfer_matrices):
    alphas = []
    mat_prod = transfer_matrices[0].copy()
    for T in transfer_matrices[1:]:
        mat_prod = np.matmul(mat_prod, T)
        norm_factor = np.average(mat_prod)
        alphas.append(norm_factor)
        mat_prod /= norm_factor
    return mat_prod, alphas


def get_F_no_interact(nucleo_arr, ind):
    
    
    
    T_all[ind] = get_transfer_matrix(ind, gamma_override = 0)
    F_no_interact


def get_F_interact():
    pass


def get_dF():
    pass


def get_P_gt():
    pass
