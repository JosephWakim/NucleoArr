"""Find conditions giving a desired average gamma for a specified chain length.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       10 Aug 2023

Usage
-----
python find_mu_var_chain_length.py <gam> <length> <mu_min> <mu_max> <mark_corr>

where <gam> is the fraction of linker lengths within interacting distance,
<length> is the number of nucleosomes on the nucleosome array, <mu_min> and
<mu_max> are the minimum and maximum values of mu to search over, and
<mark_corr> is the correlation length of the marks.
"""

import os
import sys

# Specify package root directory
notebook_directory = os.path.dirname(os.path.abspath('__file__'))
root_directory = os.path.abspath(os.path.join(notebook_directory, '..'))
os.chdir(root_directory)
sys.path.append(root_directory)

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import sliding_nucleosome.nucleo_arr as nuc
from sliding_nucleosome import mc

# Initialize physical parameters
J = np.atleast_2d([-3.92])
B = np.atleast_2d([-1.5])
mu_min = float(sys.argv[3])
mu_max = float(sys.argv[4])

# Generate a methylation sequence
n_beads = int(sys.argv[2])
frac_methyl = 0.1
methyl_corr_length = float(sys.argv[5])
marks = nuc.gen_meth(n_beads, frac_methyl, methyl_corr_length)
marks = np.atleast_2d(marks).T

# Specify the polymer
nbi = np.array([2])
linker_corr_length = 45
a = int(np.floor(15.1))
lam = -np.log(1 - 1 / linker_corr_length)

# Initialize linker lengths
linker_lengths = np.random.exponential(linker_corr_length, size=marks.shape[0])
linker_lengths = np.maximum(linker_lengths, 1.0)
linker_lengths = linker_lengths.astype(int)

# Initialize the nucleosome array
nuc_arr = nuc.NucleosomeArray(
    J=J, B=B, mu=mu, linker_lengths=linker_lengths, a=a, lam=lam,
    marks=marks, Nbi=nbi
)

# Specify parameters for linker simulation
n_snap = 20
n_steps_per_snap = 1000

target_avg_gamma = float(sys.argv[1])
mu_lower = -10.
mu_upper = -8.
rtol = 0.05

mu = mc.find_mu_for_avg_gamma(
    nuc_arr, linker_corr_length, mu_lower, mu_upper, target_avg_gamma,
    n_snap, n_steps_per_snap, rtol=rtol
)

# Initialize a nucleosome array at the desired mu condition
final_sims_directory = os.path.join("output", "final_sims")

nuc_arr = nuc.NucleosomeArray(
    J = J,
    B = B,
    mu = [mu],
    linker_lengths = linker_lengths,
    a = a,
    lam = lam,
    marks = marks,
    Nbi = nbi
)

# Simulate linker lengths
nuc_arr = mc.mc_linkers(
    nuc_arr, n_snap, n_steps_per_snap, out_dir=final_sims_directory
)
linker_lengths = nuc_arr.linker_lengths
