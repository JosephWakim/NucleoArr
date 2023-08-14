"""Run nucleosome simulations varying mark correlation length.

Usage: python vary_methyl_frac.py METHYL_FRACTION MU
where METHYL_FRACTION is the fraction of histone tails with a methylation mark
and MU is the chemical potential of the system.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       02 Aug 2023
"""
import os
import sys

# Get the absolute path to the directory containing this notebook
notebook_directory = os.path.dirname(os.path.abspath('__file__'))

# Navigate one level up to the root directory of the repository
root_directory = os.path.abspath(os.path.join(notebook_directory, '..'))

# Change the working directory to the root directory
os.chdir(root_directory)

# Append the root directory to sys.path
sys.path.append(root_directory)

# Import modules
import numpy as np
import sliding_nucleosome.nucleo_arr as nuc
from sliding_nucleosome import mc

# Initialize physical parameters
J = np.atleast_2d([-3.92])
B = np.atleast_2d([-1.5])
mu = np.array([float(sys.argv[2])])

# Generate a methylation sequence
n_beads = 1000
frac_methyl = float(sys.argv[1])
methyl_corr_length = 18.4
marks = nuc.gen_meth(n_beads, frac_methyl, methyl_corr_length)
marks = np.atleast_2d(marks).T

# Specify the polymer
gamma = np.ones(marks.shape[0])
nbi = np.array([2])
linker_corr_length = 45
linker_lengths = np.ones(marks.shape[0])
a = int(np.floor(15.1))
lam = -np.log(1 - 1 / linker_corr_length)

# Initialize the nucleosome array
nuc_arr = nuc.NucleosomeArray(
    J=J, B=B, mu=mu, linker_lengths=linker_lengths,
    a=a, lam=lam, marks=marks, Nbi=nbi
)

# Specify simulation parameters
out_dir = "output_var_methyl_frac"
n_steps = 10000
n_save = 100

# Run the simulation
mc.mc_linkers(nuc_arr, n_save, n_steps, out_dir)
