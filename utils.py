import numpy as np
from numba import njit
import logging
from scipy.special import erfc
import math
from ase import Atoms
#import cuequivariance_torch
from nequip.ase import NequIPCalculator

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

def is_float(value):
    '''Check if a string value can be converted to float'''
    try:
        float(value)
        return True
    except ValueError:
        return False

@njit
def calc_energy_numba(positions, types, particle_idx, overlap_cutoff, cutoff, box_length, sorted_distances, energy_col_0, energy_col_1, energy_col_2):
    '''Calculate total energy of a particle using tabulated PMF with periodic boundary conditions and interpolation'''
    pos = positions[particle_idx]
    particle_type = types[particle_idx]
    n_particles = len(positions)

    energy = 0.0
    min_dist_sq = 1e10

    for i in range(n_particles):
        if i == particle_idx:
            continue

        # Calculate distance with PBC using squared distances
        dx = positions[i, 0] - pos[0]
        dy = positions[i, 1] - pos[1]
        dz = positions[i, 2] - pos[2]

        # Apply periodic boundary conditions
        dx -= box_length * round(dx / box_length)
        dy -= box_length * round(dy / box_length)
        dz -= box_length * round(dz / box_length)

        dist_sq = dx*dx + dy*dy + dz*dz

        # Check cutoff (20^2 = 400)
        cutoff_sq = cutoff * cutoff
        if dist_sq < cutoff_sq:
            dist = np.sqrt(dist_sq)
            min_dist_sq = min(min_dist_sq, dist_sq)

            other_type = types[i]

            # Determine interaction type and energy column
            if particle_type == 0 and other_type == 0:
                energy_col = energy_col_0
            elif particle_type == 1 and other_type == 1:
                energy_col = energy_col_1
            else:
                energy_col = energy_col_2

            # Binary search and interpolation
            n_data = len(sorted_distances)
            left = 0
            right = n_data - 1

            while left < right - 1:
                mid = (left + right) // 2
                if sorted_distances[mid] <= dist:
                    left = mid
                else:
                    right = mid

            # Linear interpolation
            if left == n_data - 1:
                energy += energy_col[left]
            else:
                x0, x1 = sorted_distances[left], sorted_distances[right]
                y0, y1 = energy_col[left], energy_col[right]
                weight = (dist - x0) / (x1 - x0)
                energy += y0 + weight * (y1 - y0)

    # Hard-coded anti-overlap (1.5^2 = 2.25)
    overlap_cutoff_sq = overlap_cutoff * overlap_cutoff
    if min_dist_sq < overlap_cutoff_sq:
        energy = 10000.0
        
    return energy


@njit
def find_neighbors_numba(positions, pos1, cutoff, box_length):
    '''Find all particle indices within cutoff distance of given position using periodic boundary conditions'''
    cutoff_squared = cutoff**2
    neighbors = []
    for i in range(positions.shape[0]):  # number of particles
        dx = positions[i, 0] - pos1[0]
        dy = positions[i, 1] - pos1[1]
        dz = positions[i, 2] - pos1[2]

        # Apply periodic boundary conditions
        dx -= box_length * round(dx / box_length)
        dy -= box_length * round(dy / box_length)
        dz -= box_length * round(dz / box_length)

        dist_squared = dx**2 + dy**2 + dz**2
        if dist_squared < cutoff_squared:
            neighbors.append(i)
    return neighbors


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

class PMF:
    def __init__(self, types, charges, box_size, model_path):
        '''Set parameters for physics prior and load MLP'''
        # system settings
        self.types = np.array(types, dtype=np.int32)
        self.charges = np.array(charges, dtype=np.float64)
        self.box_size = box_size
        type_to_symbol = {0: 'Na', 1: 'Cl'}
        self.symbols = [type_to_symbol[t] for t in types]
        # parameters
        self.cutoff = 9.0
        self.alpha = 0.25
        self.dielectric = 73.0
        self.ke_eff = 14.39965 / self.dielectric  # eV·Å/e²

        # LJ parameters dictionary (for reference)
        self.lj_params = {
        ('Na', 'Na'): {'epsilon': 0.1, 'sigma': 2.584},
        ('Na', 'Cl'): {'epsilon': 0.1, 'sigma': 3.31},
        ('Cl', 'Cl'): {'epsilon': 0.1, 'sigma': 4.036}
        }

        # Pre-compute LJ parameter matrices for numba (indexed by type: 0=Na, 1=Cl)
        kcal_to_ev = 0.043364
        self.epsilon_matrix = np.array([
            [0.1 * kcal_to_ev, 0.1 * kcal_to_ev],  # Na-Na, Na-Cl
            [0.1 * kcal_to_ev, 0.1 * kcal_to_ev]   # Cl-Na, Cl-Cl
        ], dtype=np.float64)
        self.sigma_matrix = np.array([
            [2.584, 3.31],   # Na-Na, Na-Cl
            [3.31, 4.036]    # Cl-Na, Cl-Cl
        ], dtype=np.float64)

        # NequIP setup - LAZY LOADING (defer model loading until first use)
        self.atoms = Atoms(symbols=self.symbols, positions=np.zeros((len(self.types), 3)), cell=[box_size]*3, pbc=True)
        self.atoms.calc = None  # Ensure no calculator is attached to atoms object
        # Store model configuration instead of loading immediately
        self.model_path = model_path
        self.device = 'cuda'
        self.chemical_symbols = ['Na', 'Cl']
        self.calc = None  # Model will be loaded on first call to energies()

    def _ensure_model_loaded(self):
        '''Load the NequIP model on first use (lazy loading for multiprocessing compatibility)'''
        if self.calc is None:
            self.calc = NequIPCalculator.from_compiled_model(
                compile_path=self.model_path,
                device=self.device,
                chemical_symbols=self.chemical_symbols
            )
            self.atoms.calc = self.calc

    def energies(self, positions):
        '''Calculate total energy with physics priors and MLP'''
        # Ensure model is loaded (only happens once per process)
        self._ensure_model_loaded()

        physics_prior_energies = self.calc_prior_energies(positions)

        self.atoms.set_positions(positions)
        energy = self.atoms.get_potential_energy()

        total_energy = physics_prior_energies + energy
        # convert from eV to kcal/mol
        total_energy *= 23.0609
        return total_energy

    def calc_prior_energies(self, positions):
        '''Calculate total energy from Lennard-Jones and electrostatic interactions (numba-optimized)'''
        lj_energy = calc_lj_numba(positions, self.types, self.box_size,
                                  self.cutoff, self.epsilon_matrix, self.sigma_matrix)
        coulomb_energy = calc_coulombic_numba(positions, self.charges, self.box_size,
                                              self.cutoff, self.alpha, self.ke_eff)
        return lj_energy + coulomb_energy

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
 
