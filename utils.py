import numpy as np
from numba import njit
import logging

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

# Pre-compiled energy interpolation function
@njit
def interpolate_energy_numba(distances, sorted_distances, energies):
    '''Interpolate energies from tabulated data using binary search and linear interpolation'''
    n_distances = len(distances)
    n_data = len(sorted_distances)
    result = np.zeros(n_distances)
    
    for i in range(n_distances):
        dist = distances[i]
        
        # Binary search for the right index
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
            result[i] = energies[left]
        else:
            x0, x1 = sorted_distances[left], sorted_distances[right]
            y0, y1 = energies[left], energies[right]
            weight = (dist - x0) / (x1 - x0)
            result[i] = y0 + weight * (y1 - y0)
    
    return result

class PMF:
    def __init__(self, path):
        '''Load potential of mean force data from file and prepare for interpolation'''
        self.pmf_function = np.loadtxt(path)
        self.sorted_distances = self.pmf_function[:, 0]
        # Pre-extract energy columns for faster access
        self.energy_columns = [self.pmf_function[:, i+1] for i in range(3)]

    def energies(self, type, distances):
        '''Calculate energies for given interaction type and array of distances using interpolation'''
        distances = np.asarray(distances)
        if len(distances) == 0:
            return np.array([])
        return interpolate_energy_numba(distances, self.sorted_distances, self.energy_columns[type])

    def energy(self, type, distance):
        '''Calculate single energy value for given interaction type and distance using nearest neighbor lookup'''
        index = np.argmin(np.abs(self.pmf_function[:, 0] - distance))
        return self.pmf_function[index, type+1]

class Bias:
    def __init__(self, max_size=200, path=None, center=0, type="harmonic", force_constant=0.0, kT=0.596):
        '''Initialize bias potential for umbrella sampling with harmonic or linear bias types'''
        self.max_size = max_size
        self.type = type
        self.center = center
        self.kT = kT
        if type == "linear":
            if path is None:
                self.bias = np.zeros(max_size)
            else:
                self.bias = np.loadtxt(path)
        elif type == "harmonic":
            self.center = center
            self.force_constant = force_constant
        else:
            raise ValueError("Invalid bias type")

    def denergy(self, new, old):
        '''Calculate change in bias energy between old and new cluster sizes'''
        # Hard coding massive bias for moves that lead to clusters larger than max_size
        # Might need to move this to acceptance criteria in order to avoid overflow errors
        if new > self.max_size:
            bias = 100000
        else:
            if self.type == "harmonic":
                bias = self.force_constant/2*(new-self.center)**2 - self.force_constant/2*(old-self.center)**2
            elif self.type == "linear":
                bias = self.bias[new-1] - self.bias[old-1]
        return bias
    
    def energy(self, size):
        '''Calculate bias energy for given cluster size'''
        if size > self.max_size:
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
        new_potential = np.zeros_like(self.bias) # will likely need to change this to account for new size
        pivot_bin = np.argmax(distribution)
        n_star = distribution[pivot_bin]
        n_star_m = 1 / n_star
        
        for i in range(len(distribution)):
            if distribution[i] > 0:
                new_potential[i] = self.bias[i]  + self.kT*np.log(distribution[i] / n_star)
            else:
                new_potential[i] = self.bias[pivot_bin] + self.kT*np.log(n_star_m)
        
        # Re-shift potentials to ensure the reference state is 0kBT.
        new_potential -= new_potential[1]
        self.bias = new_potential
        return self.bias
 
