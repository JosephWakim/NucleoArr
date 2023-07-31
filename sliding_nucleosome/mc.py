"""Monte Carlo Simulator of Nucleosome Sliding and Resulting Linker Lengths.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

import os
import sys

import numpy as np
import pandas as pd

from sliding_nucleosome import binding


def get_unique_simulation_name(all_out_dir: str, sim_prefix: str) -> str:
    """Get a unique name for a simulation output directory.

    Parameters
    ----------
    all_out_dir : str
        Directory containing all simulation output directories
    sim_prefix : str
        Prefix of the simulation output directory

    Returns
    -------
    str
        Unique name for the simulation output directory
    """
    # List all simulation output directories
    all_out_dirs = os.listdir(all_out_dir)
    # Get the indices of all simulation output directories with the same prefix
    inds = [
        int(dir_.split("_")[-1]) for dir_ in all_out_dirs
        if dir_.startswith(sim_prefix)
    ]
    # Get the next index
    if len(inds) == 0:
        next_ind = 0
    else:
        next_ind = max(inds) + 1
    # Return the unique name
    return sim_prefix + "_" + str(next_ind)
