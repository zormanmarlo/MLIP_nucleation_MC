import numpy as np
from numba import njit
from ctypes import c_double, POINTER
import logging
import math

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


def _clear_mpi_env():
    '''Strip OpenMPI/PMI env vars so each forked LAMMPS worker starts MPI in standalone mode'''
    import os
    for key in list(os.environ.keys()):
        if any(key.startswith(p) for p in ('OMPI_', 'PMIX_', 'PMI_', 'ORTE_', 'I_MPI_')):
            del os.environ[key]


class PMF:
    def __init__(self, types, charges, box_size, model_path):
        '''Set parameters and build LAMMPS atom ordering'''
        self.types = np.array(types, dtype=np.int32)
        self.box_size = box_size
        self.model_path = model_path
        self.n_na = int(np.sum(self.types == 0))
        self.n_cl = int(np.sum(self.types == 1))
        # LAMMPS creates Na block then Cl block; map Python indices to that order
        self.lammps_order = np.concatenate([
            np.where(self.types == 0)[0],
            np.where(self.types == 1)[0],
        ])
        self.lmp = None  # initialized on first call
        self._pos_buffer = None  # allocated alongside lmp

    def _ensure_lammps_loaded(self):
        '''Initialize LAMMPS on first use (lazy loading for multiprocessing compatibility)'''
        if self.lmp is None:
            from lammps import lammps
            self.lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
            self._pos_buffer = np.zeros(len(self.types) * 3, dtype=np.float64)
            self._pos_ptr = self._pos_buffer.ctypes.data_as(POINTER(c_double))
            L = self.box_size
            self.lmp.commands_string(f"""
units metal
atom_style charge
boundary p p p
newton off
atom_modify map array

region box block 0 {L} 0 {L} 0 {L} units box
create_box 2 box

mass 1 22.98977
mass 2 35.453

create_atoms 1 random {self.n_na} 12345 box overlap 1.5 maxtry 1000
create_atoms 2 random {self.n_cl} 67890 box overlap 1.5 maxtry 1000

set type 1 charge  1.0
set type 2 charge -1.0

pair_style hybrid/overlay nequip coul/cut 25.0
pair_coeff * * nequip {self.model_path} Na Cl
pair_coeff * * coul/cut

dielectric 73.0

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
""")

    def energies(self, positions):
        '''Calculate total energy via LAMMPS (nequip MLIP + coul/cut prior), returns kcal/mol'''
        self._ensure_lammps_loaded()
        self._pos_buffer[:] = positions[self.lammps_order].ravel()
        self.lmp.scatter_atoms("x", 1, 3, self._pos_ptr)
        self.lmp.command("run 0 post no")
        return self.lmp.get_thermo("pe") * 23.0609

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
                # Clamp indices to the bias array bounds: `old` (and defensively `new`)
                # can exceed the table when a cached cluster has drifted past max_size.
                new_idx = min(max(new-1, 0), self.num_bins-1)
                old_idx = min(max(old-1, 0), self.num_bins-1)
                bias = self.bias[new_idx] - self.bias[old_idx]
        return bias
    
    def energy(self, size):
        '''Calculate bias energy for given cluster size'''
        if size < self.min_size or size > self.max_size:
            bias = 100000
        else:
            if self.type == "harmonic":
                bias = self.force_constant/2*(size-self.center)**2
            elif self.type == "linear":
                size_idx = min(max(size-1, 0), self.num_bins-1)
                bias = self.bias[size_idx]
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
                new_potential[i] = self.bias[i] + self.kT*np.log(distribution[i] / n_star)
            # If we have initial bias estimate but no samples, use n_star_m to update that bias
            elif self.path is not None:
                new_potential[i] = self.bias[i] + self.kT*np.log(n_star_m) 
            # Only reset bins with zero counts if we are not using an initial estimate of the bias
            elif self.path is None:
                new_potential[i] = self.bias[pivot_bin] + self.kT*np.log(n_star_m)
        
        # Re-shift potentials to ensure the reference state is 0kBT.
        new_potential -= new_potential[1]
        self.bias = new_potential
        return self.bias
 
