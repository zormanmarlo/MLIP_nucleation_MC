import numpy as np
from numba import jit

@jit(nopython=True)
def _calc_soft_sphere_energy(positions, box_length, epsilon, sigma, cutoff):
    '''Numba-optimized soft-sphere energy calculation with PBC'''
    n_atoms = len(positions)
    total_energy = 0.0

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Calculate distance with PBC (minimum image convention)
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]

            # Apply PBC
            dx = dx - box_length * np.round(dx / box_length)
            dy = dy - box_length * np.round(dy / box_length)
            dz = dz - box_length * np.round(dz / box_length)

            r_sq = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r_sq)

            # Apply soft-sphere potential with cutoff
            if r < cutoff:
                sigma_over_r = sigma / r
                total_energy += epsilon * sigma_over_r**12

    return total_energy

class SoftSpherePotential:
    '''Simple repulsive potential for testing: U(r) = ε(σ/r)^12'''

    def __init__(self, epsilon=1.0, sigma=2.0, cutoff=10.0):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def energy(self, positions, types, box_length):
        '''Calculate total soft-sphere energy with periodic boundary conditions'''
        return _calc_soft_sphere_energy(positions, box_length, self.epsilon, self.sigma, self.cutoff)
