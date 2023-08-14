"""Class representation of a nucleosome array with sliding beads.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

from typing import Optional
from scipy.special import comb
import json
import numpy as np
from wlcstat.chromo import gen_chromo_conf


class NucleosomeArray:
    """Class representation of nucleosome array and associated reader proteins.
    """
    
    def __init__(self, J, B, mu, linker_lengths, a, lam, marks, Nbi):
        """Initialize NucleosomeArray object.
        """
        # Store physical parameters
        self.J = np.atleast_2d(J)
        self.B = np.atleast_2d(B)
        self.mu = np.array(mu)
        self.linker_lengths = np.array(linker_lengths)
        self.a = a
        self.gamma = (linker_lengths <= a).astype(int)
        self.marks = np.atleast_2d(marks)
        self.Nbi = np.array(Nbi)
        self.lam = lam

        # Define unchanging physical parameters
        self.z = np.exp(-lam * a)
        self.p = 1 - np.exp(-lam)
        self.l_array = np.arange(1, a+1, dtype=int)
        p_lt = np.exp(-lam * self.l_array)
        self.p_lt = p_lt / np.sum(p_lt)

        # Count and validate the number of beads and marks
        self.n_beads = marks.shape[0]
        self.n_marks = marks.shape[1]
        assert len(Nbi) == self.n_marks, \
            "Specify a maximum binder state for each mark"

        # Get all combinations of different binder states
        self.Nr = np.prod(Nbi + 1)
        ranges = [range(Nbi_+1) for Nbi_ in Nbi]
        self.sigma_i1 = np.mgrid[tuple(ranges)]
        self.sigma_i1 = np.column_stack(
            [combo.ravel() for combo in self.sigma_i1]
        )

        # Initialize all transfer matrices
        self.get_all_transfer_matrices()

    def save(self, file_path):
        """Save nucleosome array to file.
        """
        # Convert object attributes to a JSON-formatted dictionary
        nucleosome_array_dict = {
            "J": self.J.tolist(),
            "B": self.B.tolist(),
            "mu": self.mu.tolist(),
            "linker_lengths": self.linker_lengths.tolist(),
            "a": self.a,
            "lam": self.lam,
            "marks": self.marks.tolist(),
            "Nbi": self.Nbi.tolist()
        }
        # Save dictionary to file
        with open(file_path, "w") as f:
            json.dump(nucleosome_array_dict, f)

    @classmethod
    def load(cls, file_path):
        """Load nucleosome array from file.
        """
        # Load dictionary from file
        with open(file_path, "r") as f:
            nucleosome_array_dict = json.load(f)

        # Format the mark pattern as a numpy array
        nucleosome_array_dict["marks"] = np.array(
            nucleosome_array_dict["marks"]
        ).astype(int)
        # Format linker lengths as a numpy array
        nucleosome_array_dict["linker_lengths"] = np.array(
            nucleosome_array_dict["linker_lengths"]
        ).astype(int)
        # Format Nbi as a numpy array
        nucleosome_array_dict["Nbi"] = np.array(
            nucleosome_array_dict["Nbi"]
        ).astype(int)

        # Initialize NucleosomeArray object
        return cls(**nucleosome_array_dict)

    def get_binding_energy(self, ind, sigma_ind):
        """Get the binding energy associated with binder and mark states.

        Notes
        -----
        Consider that we are arranging `Nb` binders on `Nn` sites, of which `Nm`
        are marked. We can have up to `min(Nb, Nm)` binders on marked sites. For
        each value `i` in range(`min(Nb, Nm)`), there are `comb(Nm, i)` ways to
        arrange `i` binders on marked sites and `comb(Nn-Nm, Nb-i)` ways to
        arrange the remaining `Nb-i` binders on unmarked sites. Thus, the total
        number of ways to arrange `Nb` binders on `Nn` sites, of which `Nm` are
        marked, is the sum of `comb(Nm, i) * comb(Nn-Nm, Nb-i)` for `i` in
        range(`min(Nb, Nm)`). This is equivalent to `comb(Nn, Nb)`.

        The energy of each configuration is determined by the number of binders
        on marked sites, or `i`. If there is an energy `E_b` associated with
        each binder on a marked site, then the total energy of the configuration
        is `i * E_b`. Therefore, the site partition function representing
        binding is given by `sum(comb(Nm, i) * comb(Nn-Nm, Nb-i) * exp(-i * E_b))`
        for `i` in range(`min(Nb, Nm)`).
        """
        E = 0
        # Calculate the binding energy
        for b, n_binders_ind in enumerate(sigma_ind):
            # Identify the number of associated marks on the site
            n_marks_ind = int(self.marks[ind][b])
            # Evaluate the single-site partition function
            i = np.arange(n_binders_ind + 1)
            E += -np.log(np.sum(
                comb(n_marks_ind, i) *
                comb(self.Nbi[b] - n_marks_ind, n_binders_ind - i) *
                (np.exp(-i * self.B[b, b]))
            ))
        # Calculate energies from chemical potentials and same-site interactions
        for i, binder in enumerate(sigma_ind):
            # Chemical potential
            E -= self.mu[i] * binder
            # Same-site interactions between alike binders
            E += self.J[i, i] * comb(binder, 2)
        # Same-site interactions between different binders
        for i, binder_1 in enumerate(sigma_ind[:-1]):
            for j, binder_2 in enumerate(sigma_ind[i+1:]):
                E += self.J[i, j+i+1] * binder_1 * binder_2
        return E

    def get_neighbor_interactions(self, sigma_i, sigma_ip1, gamma):
        """Get interactions between binders on adjacent sites
        """
        E = 0
        for i, binder_1 in enumerate(sigma_i):
            for j, binder_2 in enumerate(sigma_ip1):
                E += self.J[i, j] * binder_1 * binder_2 * gamma
        return E

    def get_transfer_matrix(self, ind, ind_p1, gamma_override=None):
        """Get transfer matrix at an index of the nucleosome array.
        """
        # Initialize transfer matrix and combinations of binder_states
        T = np.zeros((self.Nr, self.Nr))
        # Get the in-range parameter
        if gamma_override is None:
            gamma_ = self.gamma[ind]
        else:
            gamma_ = gamma_override
        # Populate the transfer matrix using the single-site energy function
        for row, sigma_i in enumerate(self.sigma_i1):
            for col, sigma_ip1 in enumerate(self.sigma_i1):
                E = 0
                # Binding energies (averaged over adjacent sites)
                E += (self.get_binding_energy(ind, sigma_i) +
                      self.get_binding_energy(ind_p1, sigma_ip1)) / 2
                # Neighbor interactions
                E += self.get_neighbor_interactions(sigma_i, sigma_ip1, gamma_)
                # Transfer matrix involves exponential of energy
                T[row, col] = np.exp(-E)
        return T
    
    def get_all_transfer_matrices(self):
        """Get transfer matrices for all adjacent beads.
        """
        self.T_all = np.zeros((self.Nr, self.Nr, self.n_beads))
        for ind in range(len(self.marks)):
            if ind == len(self.marks) - 1:
                ind_p1 = 0
            else:
                ind_p1 = ind + 1
            self.T_all[:, :, ind] = self.get_transfer_matrix(ind, ind_p1)
        self.T_all_1 = self.T_all.copy()


def gen_meth(n_n, f_m, l_m=0):
    """Generate a methylation profile with defined mean and correlation.

    Notes
    -----
    Function by Dr. Andrew Spakowitz.
    TODO: Talk to Andy about this.

    Parameters
    ----------
    n_n : int
        Number of nucleosomes
    f_m : float
        Probability that a tail is methylated
    l_m : float
        Correlation length of methylation

    Returns
    -------
    n_m : np.ndarray of int
        Number of methylated tails for each nucleosome
    """
    # Compute methylation probabilities
    if l_m == 0:
        lam = 0.
    else:
        lam = np.exp(-1 / l_m)
    f_u = 1 - f_m
    p_mm = lam * f_u + f_m  # P(methylated | methylated at previous site)
    p_uu = lam * f_m + f_u  # P(unmethylated | unmethylated at previous site)
    prob_tot = [(1 - f_m) ** 2, 2 * f_m * (1 - f_m), f_m ** 2]
    prob_m = [2 * (1 - f_m) / (2 - f_m), f_m / (2 - f_m)]

    # Generate the methylation profile
    n_m = np.zeros(n_n)
    n_m[0] = np.random.choice([0, 1, 2], 1, p=prob_tot)
    for i_n in range(1, n_n):
        if n_m[i_n - 1] == 0:
            if np.random.uniform(0, 1) < p_uu:
                n_m[i_n] = 0
            else:
                n_m[i_n] = np.random.choice([1, 2], 1, p=prob_m)
        else:
            if np.random.uniform(0, 1) < p_mm:
                n_m[i_n] = np.random.choice([1, 2], 1, p=prob_m)
            else:
                n_m[i_n] = 0
    return n_m


def render_nuc_arr(
    nuc_arr: NucleosomeArray, ind0: int, indf: int,
    mark_ind: Optional[int] = 0, save_path: Optional[str] = "r_poly.pdb"
):
    """Generate a PDB file representing a segment of the nucleosome array

    Parameters
    ----------
    nuc_arr : NucleosomeArray
        Nucleosome array object
    ind0 : int
        Index of the first nucleosome in the segment to be rendered
    indf : int
        One past the index of the last nucleosome in the segment to be rendered
    mark_ind : Optional[int]
        Index of the mark used to color-code beads in the PDB file (default = 0)
    save_path : Optional[str]
        Path to save the PDB file (default = "r_poly.pdb")
    """
    r, rdna1, rdna2, rn, un = gen_chromo_conf(
        nuc_arr.linker_lengths[ind0:indf]
    )
    marks_image = nuc_arr.marks[:, mark_ind].flatten()[ind0:indf]
    gen_chromo_pymol_file(
        r, rdna1, rdna2, rn, un, marks_image, filename=save_path
    )


def gen_chromo_pymol_file(
    r, rdna1, rdna2, rn, un, nm, filename='r_poly.pdb', ring=False
):
    """Generate a PDB file for a nucleosome array.

    Notes
    -----
    This function was made by Dr. Andrew Spakowitz.

    Parameters
    ----------
    r : np.ndarray
        Array of positions representing nucleosomes and linker DNA
    rdna1 : np.ndarray
        Array of positions representing first strand of DNA
    rdna2 : np.ndarray
        Array of positions representing second strand of DNA
    rn : np.ndarray
        Array of positions representing nucleosomes
    un : np.ndarray
        TODO: Check with Andy for description
    nm : np.ndarray
        Array of mark identities
    filename : Optional[str]
        Path to the PDB file to be generated (default = 'r_poly.pdb')
    ring : Optional[bool]
        Whether to generate a ring-shaped nucleosome array (default = False)
    """
    # Setup the parameters for imaging nucleosome array
    hnuc = 2.265571482035928

    # Open the file
    f = open(filename, 'w')
    atomname1 = "A1"    # Chain atom type
    atomname2 = "A2"    # Chain atom type
    atomname3 = "A3"    # Chain atom type
    atomname4 = "A4"    # Chain atom type
    atomname5 = "A5"    # Chain atom type
    atomname6 = "A6"    # Chain atom type
    resname = "SSN"     # Type of residue (UNKnown/Single Stranded Nucleotide)
    chain = "A"         # Chain identifier
    resnum = 1
    numdna = len(r[:, 0])
    numnuc = len(rn[:, 0])
    descrip = "Pseudo atom representation of DNA"
    chemicalname = "Body and ribbon spatial coordinates"

    # Write the preamble to the pymol file
    f.write('HET    %3s  %1s%4d   %5d     %-38s\n' % (resname, chain, resnum, numdna, descrip))
    f.write('HETNAM     %3s %-50s\n' % (resname, chemicalname))
    f.write('FORMUL  1   %3s    C20 N20 P21\n' % (resname))

    # Write the conformation to the pymol file
    # Define the dna centerline positions
    count = 1
    for ind in range(numdna):
        f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                (count, atomname1, resname, chain, r[ind, 0], r[ind, 1], r[ind, 2], 1.00, 1.00))
        count += 1

    # Define the dna strand 1 positions
    for ind in range(numdna):
        f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                (count, atomname2, resname, chain, rdna1[ind, 0], rdna1[ind, 1], rdna1[ind, 2], 1.00, 1.00))
        count += 1

    # Define the dna strand 2 positions
    for ind in range(numdna):
        f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                (count, atomname3, resname, chain, rdna2[ind, 0], rdna2[ind, 1], rdna2[ind, 2], 1.00, 1.00))
        count += 1

    # Define the nucleosome positions
    for ind in range(numnuc):
        rnucind = rn[ind, :] + 0.5 * hnuc *un[ind, :]
        if ind == (numnuc - 1):
            atomnamen = atomname4
        elif nm[ind] == 0:
            atomnamen = atomname4
        elif nm[ind] == 1:
            atomnamen = atomname5
        elif nm[ind] == 2:
            atomnamen = atomname6
        f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                (count, atomnamen, resname, chain, rnucind[0], rnucind[1], rnucind[2], 1.00, 1.00))
        count += 1
        rnucind = rn[ind, :] - 0.5 * hnuc *un[ind, :]
        f.write('ATOM%7d %4s %3s %1s        %8.3f%8.3f%8.3f%6.2f%6.2f           C\n' %
                (count, atomnamen, resname, chain, rnucind[0], rnucind[1], rnucind[2], 1.00, 1.00))
        count += 1

    # Define the connectivity in the chain
    # Connectivity for the center beads
    count = 1
    if ring:
        f.write('CONECT%5d%5d%5d\n' % (count, count + 1, count - 1 + numdna))
    else:
        f.write('CONECT%5d%5d\n' % (count, count + 1))
    count += 1
    for ind in range(2, numdna):
        f.write('CONECT%5d%5d%5d\n' % (count , count - 1, count + 1))
        count += 1
    if ring:
        f.write('CONECT%5d%5d%5d\n' % (count, count - 1, 1 + count - numdna))
    else:
        f.write('CONECT%5d%5d\n' % (count, count - 1))
    count += 1

    # Connectivity for the dna chain 1 beads
    if ring:
        f.write('CONECT%5d%5d%5d\n' % (count, count + 1, count - 1 + numdna))
    else:
        f.write('CONECT%5d%5d\n' % (count, count + 1))
    count += 1
    for ind in range(2, numdna):
        f.write('CONECT%5d%5d%5d\n' % (count , count - 1, count + 1))
        count += 1
    if ring:
        f.write('CONECT%5d%5d%5d\n' % (count, count - 1, 1 + count - numdna))
    else:
        f.write('CONECT%5d%5d\n' % (count, count - 1))
    count += 1

    # Connectivity for the dna chain 2 beads
    if ring:
        f.write('CONECT%5d%5d%5d\n' % (count, count + 1, count - 1 + numdna))
    else:
        f.write('CONECT%5d%5d\n' % (count, count + 1))
    count += 1
    for ind in range(2, numdna):
        f.write('CONECT%5d%5d%5d\n' % (count , count - 1, count + 1))
        count += 1
    if ring:
        f.write('CONECT%5d%5d%5d\n' % (count, count - 1, 1 + count - numdna))
    else:
        f.write('CONECT%5d%5d\n' % (count, count - 1))
    count += 1

    # Connectivity for the nucleosome positions
    for ind in range(numnuc):
        f.write('CONECT%5d%5d\n' % (count, count + 1))
        count += 1
        f.write('CONECT%5d%5d\n' % (count, count - 1))
        count += 1

    # Close the file
    f.write('END')
    f.close()
