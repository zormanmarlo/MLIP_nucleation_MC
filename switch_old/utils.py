import numpy as np
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, index, type, xyz):
        self.idx = index
        self.type = type
        self.position = np.array(xyz)
        self.cluster = None

class PMF:
    def __init__(self, path):
        self.pmf_function = np.loadtxt(path)
        # self.cut_off = self.pmf_function[-1, 0]
        #LJ_pmf = []
        #repulsive_pmf = []
#        for r in self.pmf_function[:,0]:
#            tmp = 4*20*((4/r)**12 - (4/r)**6)
#            rep_tmp = 4*3*((2.5/r)**12)
#            #tmp = 0 
#            #rep_tmp = 0
#            if tmp > 50:
#                LJ_pmf.append(50)
#            else:
#                LJ_pmf.append(tmp)
#            if rep_tmp > 50:
#                repulsive_pmf.append(50)
#            else:
#                repulsive_pmf.append(rep_tmp)
#        
#        self.pmf_function = np.concatenate((self.pmf_function, np.array(LJ_pmf).reshape(-1,1)), axis=1)
         #self.pmf_function = np.concatenate((self.pmf_function, np.array(repulsive_pmf).reshape(-1,1)), axis=1)
        #self.pmf_function[:,-1] = self.pmf_function[:,-1]*0
        #self.pmf_function[:,-2] = self.pmf_function[:,-2]*0
        self.cut_off = 20.0

    def energies(self, type, distances):
        distances = np.asarray(distances)
        sorted_distances = self.pmf_function[:, 0]

        # Find the nearest indices using binary search
        indices = np.searchsorted(sorted_distances, distances)

        # Clip indices to ensure they are within valid bounds
        indices = np.clip(indices, 1, len(sorted_distances) - 1)

        # Compare with previous index to find the closest
        left_indices = indices - 1
        right_indices = indices

        left_diff = np.abs(sorted_distances[left_indices] - distances)
        right_diff = np.abs(sorted_distances[right_indices] - distances)

        nearest_indices = np.where(left_diff <= right_diff, left_indices, right_indices)

        # Return the corresponding energy values
        return self.pmf_function[nearest_indices, type + 1]

    def energy(self, type, distance):
        index = np.argmin(np.abs(self.pmf_function[:, 0] - distance))
        return self.pmf_function[index, type+1]
    
class Bias:
    def __init__(self, target=100, path=None):
        if path is None:
            self.bias = np.zeros(target)
        else:
            self.bias = np.loadtxt(path)

    def denergy(self, new, old):
        try: 
            old_bias = self.bias[old-1]
            new_bias = self.bias[new-1]
            dE_bias = new_bias - old_bias
        except:
            dE_bias = 0 
        return dE_bias
    
    def energy(self, size):
        try:
            return self.bias[size-1]
        except:
            return 0
        
    def update(self, distribution):
        new_potential = np.zeros_like(self.bias) # will likely need to change this to account for new size
        pivot_bin = np.argmax(distribution)
        n_star = distribution[pivot_bin]
        n_star_m = 1 / n_star
        
        for i in range(len(distribution)):
            if distribution[i] > 0:
                new_potential[i] = self.bias[i]  + 0.6*np.log(distribution[i] / n_star)
            else:
                new_potential[i] = self.bias[pivot_bin] + 0.6*np.log(n_star_m)
        
        # Re-shift potentials to ensure the reference state is 0kBT.
        new_potential -= new_potential[1]
        self.bias = new_potential
        return self.bias
