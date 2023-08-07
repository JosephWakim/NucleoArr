"""Simulate reader protein binding to a chromatin fiber.

Notes
-----
This code may be used to validate that the binding isotherms are generating
results as expected.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       06Aug2023
"""

import os
from typing import Optional, List, Tuple, Dict
import numpy as np
from scipy.special import comb
from sliding_nucleosome.mc import get_unique_simulation_name


def accept_or_reject(dE: float, T: Optional[float] = 1) -> bool:
    """Accept or reject an attempted MC move.

    Parameters
    ----------
    dE : float
        Energy change associated with the attempted move
    T : Optional[float]
        Temperature of the system (default = 1)

    Returns
    -------
    bool
        True if the move is accepted, False if it is rejected
    """
    if dE <= 0:
        return True
    else:
        return np.random.rand() <= np.exp(-dE / T)


def update_state(
    accept: bool, sigma: np.ndarray, sigma_new: np.ndarray,
    inds: List[Tuple[int, int]]
) -> np.ndarray:
    """Update the state of the system to match that of an accepted move.

    Parameters
    ----------
    accept : bool
        True if the move is accepted, False if it is rejected
    sigma : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads; elements
        must be 1 or 0 for the binary model
    sigma_new : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads after the
        attempted move
    inds : List[Tuple[int, int]]
        List of tuples, each containing the bead index and the binder index
        affected by the attempted move in the form [(bead, binder), ...]

    Returns
    -------
    np.ndarray (N, Nm) of int
        Updated binding state of the system
    """
    if accept:
        for ind in inds:
            sigma[ind[0], ind[1]] = sigma_new[ind[0], ind[1]]
    return sigma


def mc_N_state_1D(
    m: np.ndarray, sigma: np.ndarray, B: np.ndarray, J: np.ndarray,
    mu: np.ndarray, n_snaps: int, n_steps_per_snap: int,
    Nbi: Optional[List[int]] = None, out_dir: Optional[str] = "output",
    out_prefix: Optional[str] = "bind_snap_"
):
    """Simulate `Nr` mark and binder states on a 1D lattice.

    Notes
    -----
    Runs a Monte Carlo simulation.

    Parameters
    ----------
    m : np.ndarray (N, Nm) of int
        Modification state for each of Nm marks on each of N beads; elements
        must be 0, 1, ..., Nr for the multistate model; here Nr is the total
        number of sites on a bead that can express a particular mark and can be
        bound by a binder
    sigma : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads; elements
        must be 0, 1, ..., Nr for the multistate model
    B : np.ndarray (Nm, Nm) of float
        Binding affinity for each combination of mark (axis 0) and binder (axis
        1)
    J : np.ndarray (Nm, Nm) of float
        For each pair of binder types (i, j), the element J[i, j] indicates the
        interaction energy gianed when binders of those types are nearby
    mu : np.ndarray (Nm,) of float
        Chemical potential of binders, biasing the binding affinity
    n_snaps : int
        Number of snapshots to save during the simulation
    n_steps_per_snap : int
        Number of Monte Carlo moves to attempt between savepoints
    Nbi : Optional[List[int]]
        Number of binders of type $i$ that can bind a single site; if None,
        then each binder type can bind a single site one time.
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
    # Save the initial state
    np.savetxt(
        os.path.join(sim_out_dir, out_prefix + "init.csv"), sigma, fmt="%i",
        delimiter=","
    )
    # Save the chemical potential
    np.savetxt(
        os.path.join(sim_out_dir, "mu.csv"), mu, fmt="%f",
        delimiter=","
    )
    # Run the simulation
    for snap in range(n_snaps):
        for ind in range(n_steps_per_snap):
            # Attempt an MC move
            sigma_new, inds = binding_move_N_state_1D(sigma, Nbi)
            # Evaluate the energy change associated with the MC move
            dE = dE_N_state_1D(m, sigma, sigma_new, inds, B, J, mu)
            # Accept or Reject the move
            accept = accept_or_reject(dE)
            # Update the state of the system
            sigma = update_state(accept, sigma, sigma_new, inds)
        # Save the current state
        np.savetxt(
            os.path.join(sim_out_dir, out_prefix + str(snap) + ".csv"), sigma,
            fmt="%i", delimiter=","
        )
        if (snap+1) % 50 == 0:
            print(f"Snapshot {snap+1} of {n_snaps} complete.")


def binding_move_N_state_1D(
    sigma: np.ndarray, Nbi: Optional[List[int]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Attempt a change in Nr-binding states along a 1D lattice.

    Notes
    -----
    Right now, this is implemented to attempt one binding move per call, but
    this could be changed to attempt multiple moves per call.

    Parameters
    ----------
    sigma : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads; elements
        must be 0, 1, ..., Nbi for the multistate model; here Nbi is the total
        number of sites on a bead that can express a particular mark and can be
        bound by a binder
    Nbi : Optional[List[int]]
        Number of binders of type $i$ that can bind a single site; if None,
        then each binder type can bind a single site one time.

    Returns
    -------
    sigma_new : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads after the
        attempted move
    inds : List[Tuple[int, int]]
        List of tuples, each containing the bead index and the binder index
        affected by the attempted move in the form [(bead, binder), ...]
    """
    if Nbi is None:
        Nbi = [1 for _ in range(sigma.shape[1])]
    # Choose a bead and a binder on that bead
    bead_ind = np.random.randint(sigma.shape[0])
    binder_ind = np.random.randint(sigma.shape[1])
    # Choose a new binding state for that binder
    sigma_new = sigma.copy()
    sigma_new[bead_ind, binder_ind] = np.random.randint(Nbi[binder_ind]+1)
    # Return new binding state and the indices of the affected bead and binder
    inds = [(bead_ind, binder_ind)]
    return sigma_new, inds


def dE_N_state_1D(
    m: np.ndarray, sigma: np.ndarray, sigma_new: np.ndarray,
    inds: Tuple[int, int], B: np.ndarray, J: np.ndarray, mu: np.ndarray
) -> float:
    """Calculate the energy change associated with a binding move.

    Notes
    -----
    Bonds are indexed by the site to the left of the bond. For example, the
    bond between the first and second beads is indexed as bond one. When we
    make an adjustment to the binding state at site i, we need to consider the
    bonds to the left and right of site i. The bond to the left of site i is
    indexed as bond i-1, and the bond to the right of site i is indexed as bond
    i.

    Parameters
    ----------
    m : np.ndarray (N, Nm) of int
        Modification state for each of Nm marks on each of N beads; elements
        must be 0, 1, ..., Nbi for the multistate model; here Nbi is the total
        number of sites on a bead that can express a particular mark and can be
        bound by a binder
    sigma : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads; elements
        must be 0, 1, ..., Nbi for the multistate model
    sigma_new : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads after the
        attempted move
    inds : List[Tuple[int, int]]
        List of tuples, each containing the bead index and the binder index
        affected by the attempted move in the form [(bead, binder), ...]
    B : np.ndarray (Nm, Nm) of float
        Binding affinity for each combination of mark (axis 0) and binder (axis
        1)
    J : np.ndarray (Nm, Nm) of float
        For each pair of binder types (i, j), the element J[i, j] indicates the
        interaction energy gianed when binders of those types are nearby
    mu : np.ndarray (Nm,) of float
        Chemical potential of binders, biasing the binding affinity

    Returns
    -------
    dE : float
        Energy change associated with the attempted move
    """
    dE = 0
    for ind in inds:

        # Evalaute energy of bond to the left of the affected site
        if ind[0] > 0:
            ind_left = [ind[0]-1, ind[1]]
        else:
            ind_left = [sigma.shape[0]-1, ind[1]]
        dE += E_bead_N_state_1D(m, sigma_new, ind_left, B, J, mu)
        dE -= E_bead_N_state_1D(m, sigma, ind_left, B, J, mu)

        # Evaluate energy of the bond to the right of the affected site
        dE += E_bead_N_state_1D(m, sigma_new, ind, B, J, mu)
        dE -= E_bead_N_state_1D(m, sigma, ind, B, J, mu)

    return dE


def E_bead_N_state_1D(
    m: np.ndarray, sigma: np.ndarray, ind: Tuple[int, int], B: np.ndarray,
    J: np.ndarray, mu: np.ndarray
) -> float:
    """Calculate the energy of a single bead in its local environment.

    Notes
    -----
    This function operates on the assumption that, on the same bead, readers
    bind marked sites before they bind unmarked sites. The readers attain the
    binding energy of all marked sites on the bead until the marked sites are
    saturated.

    Parameters
    ----------
    m : np.ndarray (N, Nm) of int
        Modification state for each of Nm marks on each of N beads; elements
        must be 0, 1, ..., Nbi for the multistate model; here Nbi is the total
        number of sites on a bead that can express a particular mark and can be
        bound by a binder
    sigma : np.ndarray (N, Nm) of int
        Binding state for each of Nm binders on each of N beads; elements
        must be 0, 1, ..., Nbi for the multistate model
    ind : Tuple[int, int]
        Tuple containing the bead index and the binder index affected by the
        attempted move in the form (bead, binder)
    B : np.ndarray (Nm, Nm) of float
        Binding affinity for each combination of mark (axis 0) and binder (axis
        1)
    J : np.ndarray (Nm, Nm) of float
        For each pair of binder types (i, j), the element J[i, j] indicates the
        interaction energy gianed when binders of those types are nearby
    mu : np.ndarray (Nm,) of float
        Chemical potential of binders, biasing the binding affinity

    Returns
    -------
    E : float
        Energy of the bead in its local environment
    """
    # To support interpretability, let's name some indices
    bead = ind[0]
    n_beads = sigma.shape[0]
    if bead < n_beads-1:
        bead_right = bead + 1
    else:
        bead_right = 0
    n_binders = sigma.shape[1]
    E_bind = 0
    E_local_int = 0
    E_neigh_int = 0

    # Binding energy
    for i in range(n_binders):
        E_bind += np.min([sigma[bead, i], m[bead, i]]) * B[i, i]
        E_bind -= sigma[bead, i] * mu[i]

    # Same-site, same-binder interactions
    for binder in range(n_binders):
        E_local_int += comb(sigma[bead, binder], 2) * J[binder, binder]

    # Same-site, different-binder interactions
    for binder_1 in range(n_binders - 1):
        for binder_2 in range(binder_1 + 1, n_binders):
            E_local_int += \
                sigma[bead, binder_1] * sigma[bead, binder_2] * \
                J[binder_1, binder_2]

    # Different-site interactions
    for binder_1 in range(n_binders):
        for binder_2 in range(n_binders):
            E_neigh_int += \
                sigma[bead, binder_1] * sigma[bead_right, binder_2] * \
                J[binder_1, binder_2]

    return E_bind + E_local_int + E_neigh_int