"""Monte Carlo Simulator of Nucleosome Sliding and Resulting Linker Lengths.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

import os
from typing import Optional
import numpy as np
import sliding_nucleosome.nucleo_arr as nuc
import sliding_nucleosome.linkers as link


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


def mc_linkers(
    nuc_arr: nuc.NucleosomeArray, n_snaps: int, n_steps_per_snap: int,
    out_dir: Optional[str] = "output", out_prefix: Optional[str] = "snap_"
):
    """Sample new linker lengths along a nucleosome array using Monte Carlo.

    Parameters
    ----------
    nuc_arr : nuc.NucleosomeArray
        Nucleosome array for which new linker lengths are to be sampled
    n_snaps : int
        Number of snapshots to save during the simulation
    n_steps_per_snap : int
        Number of Monte Carlo moves to attempt between savepoints
    out_dir : Optional[str]
        Output directory into which snapshots will be saved (default = "Output")
    out_prefix : Optional[str]
        Prefix of each snapshot filename in the output directory, to be
        followed by the snapshot number and the file extension (default =
        "snap_")
    """
    # Make the output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Make simulation output directory
    sim_out_dir = get_unique_simulation_name(out_dir, "sim")
    sim_out_dir = os.path.join(out_dir, sim_out_dir)
    os.makedirs(sim_out_dir)
    # Save the initial state of the nucleosome array
    nuc_arr.save(os.path.join(sim_out_dir, "snap_init.json"))
    # Run the simulation
    for snap in range(n_snaps):
        for ind in range(n_steps_per_snap):
            # Select a random linker index
            link_ind = np.random.randint(0, nuc_arr.n_beads)
            # Sample a new linker length
            link.linker_move(nuc_arr, link_ind)
        # Save the current state
        nuc_arr.save(
            os.path.join(sim_out_dir, out_prefix + str(snap) + ".json")
        )
        if (snap+1) % 50 == 0:
            print(f"Snapshot {snap+1} of {n_snaps} complete.")
