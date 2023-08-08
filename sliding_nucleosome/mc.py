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
from binding_model.theory_bind import find_mu


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
) -> nuc.NucleosomeArray:
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
    return nuc_arr


def bind_slide(
    nuc_arr: nuc.NucleosomeArray, mu_lower: float, mu_upper: float,
    theta_target: float, n_snap: int, n_steps_per_snap: int,
    binder_ind: Optional[int] = 0, iter_: Optional[int] = 0,
    max_iters: Optional[int] = 10, rtol_bind: Optional[float] = 0.01,
    rtol_mu: Optional[float] = 0.01, prev_mu: Optional[float] = 1e99,
    rtol_gamma: Optional[float] = 0.01, prev_avg_gamma: Optional[float] = 1e99
) -> nuc.NucleosomeArray:
    """Iterative between binding model and linker model.

    Notes
    -----
    This function iterates on the binding model and the linker length model to
    identify a chemical potential that yields the desired site occupancy and
    has an equilibrium distribution of linker lengths that matches associated
    theory. The binding model is run first, followed by the linker length
    model. With the linker lengths of the nucleosome array updated, the function
    is called recursively until the chemical potential and linker lengths
    converge.

    The function will be updated in the future to work with multiple binders.

    Parameters
    ----------
    nuc_arr : nuc.NucleosomeArray
        Nucleosome array for which new linker lengths and chemical potentials
        are to be sampled
    mu_lower, mu_upper : float
        Lower and upper bounds on the chemical potential thought to contain
        the chemical potential for the desired site occupancy
    theta_target : float
        Target site occupancy
    n_snap : int
        Number of snapshots to save during the linker simulation
    n_steps_per_snap : int
        Number of Monte Carlo moves to attempt between savepoints during the
        linker simulation
    binder_ind : Optional[int]
        Index of the binder to be evaluated by the binding model (default = 0)
    iter_ : Optional[int]
        Current iteration number (default = 0)
    max_iters : Optional[int]
        Maximum number of iterations (default = 10)
    rtol_bind : Optional[float]
        Relative tolerance for the site occupancy (default = 0.01)
    rtol_mu : Optional[float]
        Relative tolerance for the chemical potential (default = 0.01)
    prev_mu : Optional[float]
        Previous chemical potential (default = 1e99)
    rtol_gamma : Optional[float]
        Relative tolerance for the equilibrium distribution of linker lengths
        (default = 0.01)
    prev_avg_gamma: Optional[float]
        Fraction of the previous iteration's linker lengths that were within
        the interaction range (default = 1e99)

    Returns
    -------
    nuc_arr : nuc.NucleosomeArray
        Nucleosome array with new linker lengths and chemical potentials
        that yield the desired site occupancy and have an equilibrium
        distribution of linker lengths that matches associated theory
    """
    # Update iteration
    iter_ += 1
    print(f"Starting iteration {iter_} of {max_iters}.")

    # Update the chemical potential
    mu_iter = find_mu(
        nuc_arr, mu_lower, mu_upper, theta_target, binder_ind, rtol=rtol_bind
    )

    # Update the linker length distribution
    nuc_arr.mu[binder_ind] = mu_iter
    nuc_arr = mc_linkers(nuc_arr, n_snap, n_steps_per_snap)
    gamma_iter = nuc_arr.gamma[binder_ind]
    avg_gamma = np.average(gamma_iter)

    # Base Cases
    if iter_ >= max_iters:
        print(f"Maximum number of iterations reached.")
        return nuc_arr
    change_in_range_frac = (
        (avg_gamma - prev_avg_gamma) / prev_avg_gamma
    )
    change_mu = (mu_iter - prev_mu) / prev_mu
    converged = (np.abs(change_in_range_frac) < rtol_gamma) and \
                (np.abs(change_mu) < rtol_mu)
    if converged:
        print("Bind and Slide Converged!")
        return nuc_arr

    # Recursive Case
    else:
        # Reiterate with the updated linker length distribution
        return bind_slide(
            nuc_arr, mu_lower, mu_upper, theta_target, n_snap,
            n_steps_per_snap, binder_ind, iter_, max_iters, rtol_bind,
            rtol_mu, mu_iter, rtol_gamma, avg_gamma
        )
