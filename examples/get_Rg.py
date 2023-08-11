"""Identify and characterize clusters in a nucleosome array.

By:         Joseph Wakim, Andrew Spakowitz
Group:      Spakowitz Lab
Date:       11 August 2023

This script identifies clusters of nucleosomes based on their linker lengths.
A cluster is defined by a continuous sequence of nucleosomes with linker
lengths smaller than the cutoff interaction distance `a`.

Usage: python get_Rg.py <out_path>
where <out_path> is the path to the sliding nucleosome simulation output
directory.
"""

import os
import sys

# Get the absolute path of the notebook's directory
notebook_directory = os.path.dirname(os.path.abspath('__file__'))
root_directory = os.path.abspath(os.path.join(notebook_directory, '..'))
os.chdir(root_directory)
sys.path.append(root_directory)

# Import modules
from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sliding_nucleosome.nucleo_arr as nuc
from wlcstat.chromo import gen_chromo_conf

def radius_of_gyration(coordinates):
    """Calculate the radius of gyration for a set of coordinates.
    """
    center_of_mass = np.mean(coordinates, axis=0)
    displacement_vectors = coordinates - center_of_mass
    squared_distances = np.sum(displacement_vectors ** 2, axis=1)
    rg_squared = np.mean(squared_distances)
    rg = np.sqrt(rg_squared)
    return rg

# Specify the output directory
out_path = sys.argv[1]

# Identify the latest snapshot
snapshots = [f for f in os.listdir(out_path) if f.startswith('snap') and f.endswith('.json')]
snapshots = [f for f in snapshots if f != 'snap_init.json']

# Load the latest snapshot
snapshot_inds = [int(f.split('_')[1].split('.')[0]) for f in snapshots]
latest_snapshot = snapshots[np.argmax(snapshot_inds)]

# Load the snapshot
snap_path = os.path.join(out_path, latest_snapshot)
nuc_arr = nuc.NucleosomeArray.load(snap_path)

## Identify clusters from "in-range" parameter gamma
# Specify a minumum cluster count (this is arbitrary)
min_count = 4
# Identify linker lengths within interaction distance
gamma = nuc_arr.gamma
# Group `gamma` into sets of equivalent values
adjacent_sets = groupby(gamma)
# Characterize the sets
set_types = {}
set_sizes = {}
bead_set_map = {}
set_bead_map = {}
index = 0
for i, (key, group) in enumerate(adjacent_sets):
    set_types[i] = key
    set_size = len(list(group))
    set_sizes[i] = set_size
    set_bead_map[i] = []
    for bead in range(index, index+set_size):
        bead_set_map[bead] = i
        set_bead_map[i].append(bead)
    index += set_size
# Identify clusters (sets of type `1` with size >= 4)
n_sets = len(list(set_types.keys()))
clusters = [
    i for i in range(n_sets)
    if (set_types[i] == 1) and (set_sizes[i] >= min_count)
]
# Identify nucleosomes in each cluster
cluster_inds = {cluster: set_bead_map[cluster] for cluster in clusters}

# Specify the output file for Rg data
save_name = "Rg_dist.csv"
save_path = os.path.join(out_dir, save_name)

## Generate a representative distribution for radius of gyration
n_realizations = 1000
Rg_all = []
for i in range(n_realizations):
    # Generate a configuration for the nucleosome array
    _, _, _, r_nucleo, _ = gen_chromo_conf(nuc_arr.linker_lengths)
    # Identify coordinates for each cluster
    cluster_r = {
        cluster: r_nucleo[inds, :] for cluster, inds in cluster_inds.items()
    }
    # Calculate the radius of gyration for each cluster
    cluster_Rg = {
        cluster: radius_of_gyration(coords)
        for cluster, coords in cluster_r.items()
    }
    # Store the radii of gyration
    Rg_all += list(cluster_Rg.values())
    R2_arr = np.array(Rg_all)
    # Save update (inefficient, but it protects against interrupted jobs)
    with open(save_path, 'w') as f:
        f.write('Rg (nm)\n')
        np.savetxt(f, R2_arr)
