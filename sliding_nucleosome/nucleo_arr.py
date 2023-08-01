"""Class representation of a nucleosome array with sliding beads.

By:         Joseph Wakim
Group:      Spakowitz Lab
Date:       31 July 2023
"""

from math import comb
import json
import numpy as np


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
        self.l_array = np.arange(1, a, dtype=int)
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
        self.T_all_1 = self.T_all.copy()

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
        )

        # Initialize NucleosomeArray object
        return cls(**nucleosome_array_dict)

    def get_transfer_matrix(self, ind, gamma_override=None):
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
                # Binding energies
                for i, binder in enumerate(sigma_i):
                    E += self.B[i, i] * np.min([binder, self.marks[ind][i]])
                    # Chemical potential
                    E -= self.mu[i] * binder
                # Neighbor interactions
                for i, binder_1 in enumerate(sigma_i):
                    for j, binder_2 in enumerate(sigma_ip1):
                        E += self.J[i, j] * binder_1 * binder_2 * gamma_
                # Same-site interactions between alike binders
                for i, binder in enumerate(sigma_i):
                    E += self.J[i, i] * comb(binder, 2)
                # Same-site interactions between different binders
                for i, binder_1 in enumerate(sigma_i[:-1]):
                    for j, binder_2 in enumerate(sigma_i[i+1:]):
                        E += self.J[i, j+i+1] * binder_1 * binder_2
                # Transfer matrix involves exponential of energy
                T[row, col] = np.exp(-E)
                # Scale for degeneracy
                degeneracy = 1
                for i, binder in enumerate(sigma_i):
                    degeneracy *= comb(2, binder)
                for i, binder in enumerate(sigma_ip1):
                    degeneracy *= comb(2, binder)
                if degeneracy > 1:
                    T[row, col] *= np.log(degeneracy)
        return T
    
    def get_all_transfer_matrices(self):
        """Get transfer matrices for all adjacent beads.
        """
        self.T_all = np.zeros((self.Nr, self.Nr, self.n_beads))
        for ind in range(len(self.marks)):
            self.T_all[:, :, ind] = self.get_transfer_matrix(ind)
