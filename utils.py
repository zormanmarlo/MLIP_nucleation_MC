import numpy as np
from numba import njit
import logging
from scipy.special import erfc
import math
from lammps import lammps
from ctypes import c_double

def setup_logger():
    '''Initialize logger for Monte Carlo simulation with appropriate formatting and handlers'''
    logger = logging.getLogger('monte_carlo')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def euler_to_rotation_matrix(phi, theta, psi):
    '''Convert Euler angles (ZYZ convention) to rotation matrix: r_lab = R @ r_body'''
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)

    R = np.array([
        [cos_phi*cos_theta*cos_psi - sin_phi*sin_psi, -cos_phi*cos_theta*sin_psi - sin_phi*cos_psi,  cos_phi*sin_theta],
        [sin_phi*cos_theta*cos_psi + cos_phi*sin_psi, -sin_phi*cos_theta*sin_psi + cos_phi*cos_psi,  sin_phi*sin_theta],
        [-sin_theta*cos_psi,                            sin_theta*sin_psi,                            cos_theta]
    ])
    return R

def random_euler_angles():
    '''Generate uniformly random Euler angles for SO(3) rotation (ZYZ convention)'''
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.arccos(np.random.uniform(-1, 1))
    psi = np.random.uniform(0, 2*np.pi)
    return phi, theta, psi

def random_rotation_matrix():
    '''Generate uniformly random 3D rotation matrix using Euler angles (ZYZ convention)'''
    phi, theta, psi = random_euler_angles()
    return euler_to_rotation_matrix(phi, theta, psi)

@njit(cache=False)
def calc_coulombic_numba(positions, charges, box_size, cutoff, alpha, ke_eff):
    '''Calculate electrostatic energy using Wolf summation with PBC (numba-optimized).

    Args:
        positions: Array of particle positions (N x 3)
        charges: Array of particle charges (N,)
        box_size: Size of periodic box
        cutoff: Cutoff distance for interactions
        alpha: Wolf summation damping parameter
        ke_eff: Effective Coulomb constant (ke / dielectric)

    Returns:
        Total Coulombic energy in eV
    '''
    n_atoms = len(positions)
    total_energy = 0.0
    alpha_r_cut = alpha * cutoff

    # Pre-compute erfc(alpha * r_cut) using math.erfc (numba compatible)
    erfc_alpha_r_cut = math.erfc(alpha_r_cut)

    # Calculate pairwise electrostatic energy using Wolf summation
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Calculate distance with PBC (inline for speed)
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]

            # Apply periodic boundary conditions
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)
            dz -= box_size * round(dz / box_size)

            r = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Skip if beyond cutoff
            if r >= cutoff:
                continue

            # Handle overlapping particles (shouldn't happen in well-behaved simulation)
            if r < 0.01:  # particles closer than 0.01 Angstrom
                return 1e10  # Return very high energy to reject this configuration

            qi = charges[i]
            qj = charges[j]

            # Wolf potential: U = ke * qi * qj * [erfc(α*r)/r - erfc(α*R_c)/R_c]
            alpha_r = alpha * r
            erfc_alpha_r = math.erfc(alpha_r)

            # Wolf pair energy
            pair_energy = ke_eff * qi * qj * (
                erfc_alpha_r / r - erfc_alpha_r_cut / cutoff
            )
            total_energy += pair_energy

    # Add Wolf self-energy correction for each atom
    # Self-energy correction: -ke * qi^2 * alpha/sqrt(pi)
    sqrt_pi = 1.7724538509055159  # math.sqrt(math.pi) precomputed
    for i in range(n_atoms):
        qi = charges[i]
        self_energy = -ke_eff * qi * qi * alpha / sqrt_pi
        total_energy += self_energy

    return total_energy


@njit(cache=False)
def calc_lj_numba(positions, types, box_size, cutoff, epsilon_matrix, sigma_matrix):
    '''Calculate Lennard-Jones interaction energy with PBC (numba-optimized).

    Args:
        positions: Array of particle positions (N x 3)
        types: Array of particle types (N,) - integer indices
        box_size: Size of periodic box
        cutoff: Cutoff distance for interactions
        epsilon_matrix: 2D array of epsilon values indexed by [type_i, type_j] in eV
        sigma_matrix: 2D array of sigma values indexed by [type_i, type_j] in Angstroms

    Returns:
        Total LJ energy in eV
    '''
    n_atoms = len(positions)
    total_energy = 0.0

    # Calculate energy for all pairs
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Get atom types for this pair
            type_i = types[i]
            type_j = types[j]

            # Get LJ parameters from matrices
            epsilon = epsilon_matrix[type_i, type_j]
            sigma = sigma_matrix[type_i, type_j]

            # Calculate distance with PBC (inline for speed)
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]

            # Apply periodic boundary conditions
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)
            dz -= box_size * round(dz / box_size)

            r = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Skip if beyond cutoff
            if r >= cutoff:
                continue

            # Handle overlapping particles (shouldn't happen in well-behaved simulation)
            if r < 0.01:  # particles closer than 0.01 Angstrom
                return 1e10  # Return very high energy to reject this configuration

            # Calculate LJ energy: U_LJ = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
            sigma_over_r = sigma / r
            sigma6 = sigma_over_r * sigma_over_r * sigma_over_r
            sigma6 = sigma6 * sigma6  # sigma^6
            sigma12 = sigma6 * sigma6  # sigma^12

            # Energy for this pair
            pair_energy = 4.0 * epsilon * (sigma12 - sigma6)
            total_energy += pair_energy

    return total_energy


class PMFLammps:
    def __init__(self, types, charges, box_size, model_path):
        '''Set up LAMMPS-based energy evaluator with allegro + PPPM electrostatics'''
        from lammps import lammps
        self.types = np.array(types, dtype=np.int32)
        self.charges = np.array(charges, dtype=np.float64)
        self.box_size = box_size
        self.model_path = model_path
        self.lmp = None  # lazy load for multiprocessing compatibility

        self.n_type1 = int(np.sum(self.types == 0))  # Ca
        self.n_type2 = int(np.sum(self.types == 1))  # C
        self.n_type3 = int(np.sum(self.types == 2))  # O

        # lammps_order[i] = Python index of LAMMPS atom i
        # LAMMPS creates type-1 (Ca), then type-2 (C), then type-3 (O)
        self.lammps_order = np.concatenate([
            np.where(self.types == 0)[0],
            np.where(self.types == 1)[0],
            np.where(self.types == 2)[0]
        ])

    def _init_lammps(self):
        '''Initialize LAMMPS instance with allegro + coul/long pair style'''
        self.lmp = lammps()
        L = self.box_size
        self.lmp.commands_string(f"""
units metal
atom_style charge
boundary p p p
atom_modify map array

region box block 0 {L} 0 {L} 0 {L} units box
create_box 3 box

mass 1 40.078   # Ca
mass 2 12.011   # C
mass 3 15.999   # O

create_atoms 1 random {self.n_type1} 12345 box overlap 2.0 maxtry 1000
create_atoms 2 random {self.n_type2} 67890 box overlap 2.0 maxtry 1000
create_atoms 3 random {self.n_type3} 11223 box overlap 2.0 maxtry 1000

set type 1 charge  2.0
set type 2 charge  1.423285
set type 3 charge -1.141095 

pair_style hybrid/overlay allegro coul/long 9.0
pair_coeff * * allegro {self.model_path} Ca C O
pair_coeff * * coul/long
kspace_style pppm 1e-6

dielectric 73.0

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
""")

    def energies(self, positions):
        '''Calculate total energy via LAMMPS; returns kcal/mol'''
        if self.lmp is None:
            self._init_lammps()

        lammps_pos = positions[self.lammps_order].flatten()
        x = (c_double * len(lammps_pos))(*lammps_pos)
        self.lmp.scatter_atoms("x", 1, 3, x)
        self.lmp.command("run 0 post no")
        pe = self.lmp.get_thermo("pe")  # eV (units metal)
        return pe * 23.0609  # convert to kcal/mol


class Bias:
    def __init__(self, max_size=200, min_size=0, path=None, center=0, type="harmonic", force_constant=0.0, kT=0.596):
        '''Initialize bias potential for umbrella sampling with harmonic or linear bias types'''
        self.max_size = max_size
        self.min_size = min_size
        self.type = type
        self.center = center
        self.kT = kT
        self.path = path
        if type == "linear":
            if path is None:
                self.bias = np.zeros(max_size)
            else:
                self.bias = np.loadtxt(path)
            self.num_bins = len(self.bias)
        elif type == "harmonic":
            self.center = center
            self.force_constant = force_constant
        else:
            raise ValueError("Invalid bias type")

    def denergy(self, new, old):
        '''Calculate change in bias energy between old and new cluster sizes'''
        # Hard coding massive bias for moves that lead to clusters smaller than min_size or larger than max_size
        # Might need to move this to acceptance criteria in order to avoid overflow errors
        if new < self.min_size or new > self.max_size:
            bias = 100000
        else:
            if self.type == "harmonic":
                bias = self.force_constant/2*(new-self.center)**2 - self.force_constant/2*(old-self.center)**2
            elif self.type == "linear":
                bias = self.bias[new-1] - self.bias[old-1]
        return bias
    
    def energy(self, size):
        '''Calculate bias energy for given cluster size'''
        if size < self.min_size or size > self.max_size:
            bias = 100000
        else:
            if self.type == "harmonic":
                bias = self.force_constant/2*(size-self.center)**2
            elif self.type == "linear":
                bias = self.bias[size-1]
        return bias
    
    # Update bias for adaptive US
    def update(self, distribution):
        '''Update bias potential for adaptive umbrella sampling based on cluster size distribution'''
        new_potential = self.bias.copy()
        pivot_bin = np.argmax(distribution)
        n_star = distribution[pivot_bin]
        n_star_m = 1 / n_star
        
        for i in range(len(distribution)):
            if distribution[i] > 0:
                new_potential[i] = self.bias[i]  + self.kT*np.log(distribution[i] / n_star)
            # Only reset bins with zero counts if we are not using an initial estimate of the bias
            elif self.path is None:
                new_potential[i] = self.bias[pivot_bin] + self.kT*np.log(n_star_m)
        
        # Re-shift potentials to ensure the reference state is 0kBT.
        new_potential -= new_potential[1]
        self.bias = new_potential
        return self.bias
 
