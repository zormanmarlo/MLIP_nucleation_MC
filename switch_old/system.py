import random
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from utils import PMF, Particle, Bias

class System:
    def __init__(self, l, n, s=11, pmf_path="potentials/dft_prot.txt", bias_path=None, target_max=100):
        #self.box_length = 123.2
        self.box_length = l
        self.num_particles = n
        self.particles = []
        self.surf_particles = []

        self.seed = s
        np.random.seed(self.seed)

        self.pmf = PMF(pmf_path)
        self.cut_off = self.pmf.cut_off
        self.bias = Bias(path=bias_path, target=target_max)

        self.target_clust_idx = []
        self.cluster_sizes = []
        self.tmp_target_clust_idx = self.target_clust_idx.copy()

        self.energy = 0.0
        self.energy_log = []
        self.bias_energy = 0.0
        self.bias_energy_log = []
        
        self.rejected_rates = [0,0,0,0,0]
        self.attempts = [0,0,0,0,0]

        self.kT = 0.6
        self.clust_cutoff = 3.9
        self.low_cutoff = 2.7
        self.vol_ratio = (4.0/3.0 * np.pi * (self.clust_cutoff**3)) / (self.box_length**3)
        self.Vin = 4.0/3.0 * np.pi * (self.clust_cutoff**3)
        self.Vout = self.box_length**3

        self.max_displacement = 4.2

    def init(self, input_path=None):
        if input_path is not None:
            with open(input_path, 'r') as f:
                for i, line in enumerate(f.readlines()[2:]):
                    tmp = line.split()
                    x, y, z = float(tmp[1]), float(tmp[2]), float(tmp[3])
                    assign_value = lambda atom: 0 if atom == "H" else 1 if atom == "O" else None
                    part_type = assign_value(str(tmp[0]))
                    self.particles.append(Particle(i, part_type, np.array([x, y, z])))
        else:
            n_species = int(self.num_particles/2)
            # Calciums
            for i in range(n_species):
                position = np.round(np.random.rand(3) * self.box_length, 3)
                position[2] = np.round(np.random.rand(1) * (self.box_length-22))+11
                self.particles.append(Particle(i, 0, position))
            # Carbonates
            for i in range(n_species):
                i += n_species
                position = np.round(np.random.rand(3) * self.box_length, 3)
                position[2] = np.round(np.random.rand(1) * (self.box_length-22))+11
                self.particles.append(Particle(i, 1, position))
        
        with open("inputs/FD31_nonideal.xyz", "r") as f:
            for i, line in enumerate(f.readlines()[2:]):
                tmp = line.split()
                x, y, z = float(tmp[1])+(self.box_length/2), float(tmp[2])+(self.box_length/2), float(10.0)
                #if tmp[0] == "O":
                #    part_type = 1
                #elif tmp[0] == "H":
                #    part_type = 0
                #else:
                part_type = 2
                self.surf_particles.append(Particle(i, part_type, np.array([x, y, z])))
            
            # Shift center of surface to box center of mass
            com = np.mean([p.position for p in self.surf_particles], axis=0)
            shift = np.array([self.box_length/2-com[0], self.box_length/2-com[1], 0])
            for particle in self.surf_particles:
                particle.position[0] += shift[0]
                particle.position[1] += shift[1]
        

        self.num_surf_particles = len(self.surf_particles)
        self.energy = self.calc_full_energy()
    
    def find_clusters(self):
        # Convert positions to a NumPy array for vectorized operations
        positions = np.array([p.position for p in self.particles])

        # Compute pairwise distances using broadcasting
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        diff = diff - self.box_length * np.round(diff / self.box_length)
        dist_matrix = np.linalg.norm(diff, axis=2)
        # dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

        # Create adjacency matrix based on the cutoff
        adjacency_matrix = (dist_matrix < self.clust_cutoff).astype(int)
        adjacency_matrix_sparse = csr_matrix(adjacency_matrix)

        # Compute connected components
        n_clusters, labels = connected_components(csgraph=adjacency_matrix_sparse, directed=False)
        
        # Compute cluster sizes
        self.cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        self.target_clust_idx = np.where(labels == labels[0])[0].tolist()

        return self.cluster_sizes, self.target_clust_idx
        
    
    def find_cluster_around_target(self, target_idx=0):
        # Compute distance matrix (only once for efficiency)
        pos = np.array([p.position for p in self.particles])
        dist_matrix = squareform(pdist(pos))  # Compute the distance matrix for all particles

        # Create an adjacency matrix where True indicates a distance less than cutoff
        adj_matrix = dist_matrix < self.clust_cutoff

        # Now perform a graph search (BFS or DFS) starting from target_idx
        visited = set()
        queue = [target_idx]
        target_clust_idx = []

        while queue:
            current_idx = queue.pop(0)  # BFS, use pop() for DFS
            if current_idx not in visited:
                visited.add(current_idx)
                target_clust_idx.append(current_idx)
                
                # Find all connected nodes (particles within the cutoff)
                neighbors = np.where(adj_matrix[current_idx])[0]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)

        self.target_clust_idx = target_clust_idx
        return self.target_clust_idx
        
    def check_in(self, particle_idx):
        for clust_idx in self.target_clust_idx:
            if clust_idx != particle_idx:
                distance = self.calc_dist(self.particles[particle_idx].position, self.particles[clust_idx].position)
                if distance < self.clust_cutoff:
                    return True
        return False
        
    def calc_in(self, particle):
        positions = np.array([p.position for p in self.particles])
        particle_position = particle.position
        
        distances = positions - particle_position
        distances = distances - self.box_length * np.round(distances / self.box_length)
        distances = np.linalg.norm(distances, axis=1)      
        close_indices = np.where((distances < self.clust_cutoff) & (np.arange(len(self.particles)) != particle.idx))[0]
        
        Nin = len(close_indices)
        Nin_idx = [self.particles[i].idx for i in close_indices]
        
        return Nin, Nin_idx
    
    def calc_energy(self, particle_idx):
        particle1 = self.particles[particle_idx]
        pos1 = particle1.position
        type1 = particle1.type

        # Extract positions and types as numpy arrays
        positions = np.array([p.position for p in self.particles])
        surf_positions = np.array([p.position for p in self.surf_particles])
        positions = np.concatenate((positions, surf_positions), axis=0)
        types = np.array([p.type for p in self.particles])
        surf_types = np.array([p.type for p in self.surf_particles])
        types = np.concatenate((types, surf_types), axis=0)

        # Calculate distance vector with periodic boundary conditions
        distances = positions - pos1
        distances = distances - self.box_length * np.round(distances / self.box_length)
        distances = np.linalg.norm(distances, axis=1)

        # Mask for particles within cut-off distance (excluding particle1 itself)
        within_cutoff = (distances < self.cut_off) & (distances > 0.0)

        # Apply the cutoff mask
        cutoff_distances = distances[within_cutoff]
        cutoff_types = types[within_cutoff]

        # Combined mask for type-specific interactions
        type_masks = [
            (cutoff_types == 0) & (type1 == 0),                         # Type 0-0 interactions Ca-Ca 
            (cutoff_types == 1) & (type1 == 1),                         # Type 1-1 interactions CO3-CO3
            (cutoff_types != type1) & (cutoff_types != 2),              # Type 0-1 interactions Ca-CO3 / CO3-Ca
            (cutoff_types == 2) & (type1 == 0),                         # Type 0-2 interactions Ca-Protein / Protein-Ca
            (cutoff_types == 2) & (type1 == 1),                         # Type 1-2 interactions CO3-Protein / Protein-CO3

        ]

        # Pre-compute energies for each type of interaction
        dE = sum(
            np.sum(self.pmf.energies(i, cutoff_distances[mask]))
            for i, mask in enumerate(type_masks)
            if np.any(mask)
        )

        return dE
    
    def calc_full_energy(self):
        self.energy = 0.0
        for i, particle1 in enumerate(self.particles):
            for j, particle2 in enumerate(self.particles):
                if i < j:
                    distance = self.calc_dist(particle1.position, particle2.position)
                    if distance < self.cut_off:
                        if particle1.type == 0 and particle2.type == 0:
                            self.energy += self.pmf.energy(0, distance)
                        if particle1.type == 1 and particle2.type == 1:
                            self.energy += self.pmf.energy(1, distance)
                        if (particle1.type == 0 and particle2.type == 1) or (particle1.type == 1 and particle2.type == 0):
                            self.energy += self.pmf.energy(2, distance)
                        else:
                            self.energy += self.pmf.energy(3, distance)
        
        tmp_energy = 0.0
        for i, particle in enumerate(self.particles):
            for j, surf_particle in enumerate(self.surf_particles):
                distance = self.calc_dist(particle.position, surf_particle.position)
                if distance < self.cut_off:
                    if particle.type == 0 and surf_particle.type == 0:
                        tmp_energy += self.pmf.energy(0, distance)
                    if particle.type == 1 and surf_particle.type == 1:
                        tmp_energy += self.pmf.energy(1, distance)
                    if (particle.type == 0 and surf_particle.type == 1) or (particle.type == 1 and surf_particle.type == 0):
                        tmp_energy += self.pmf.energy(2, distance)
                    else:
                        tmp_energy += self.pmf.energy(3, distance)
        return self.energy

    def calc_dist(self, pos1, pos2):
        dist_vec = np.abs(pos1 - pos2)
        dist_vec = dist_vec - self.box_length * np.round(dist_vec / self.box_length)
        return np.linalg.norm(dist_vec)
    
    def translation(self, particle_idx):
        self.attempts[0] += 1

        old_pos = self.particles[particle_idx].position
        displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)
        
        while (np.sum(displacement**2) > self.max_displacement**2) or ((old_pos[2] + displacement[2]) % self.box_length < 11) or ((old_pos[2] + displacement[2]) % self.box_length > self.box_length-11):
            # while ((old_pos[2] + displacement[2]) % self.box_length < 11) or ((old_pos[2] + displacement[2]) % self.box_length > self.box_length-11):
                displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)

        new_pos = (old_pos + displacement) % (self.box_length)

        old_energy = self.calc_energy(particle_idx)
        self.particles[particle_idx].position = new_pos

        # self.tmp_target_clust_idx = self.target_clust_idx.copy()
        # self.target_clust_idx = self.find_cluster_around_target()
        # bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))
        bias_energy = 0.0

        new_energy = self.calc_energy(particle_idx)
        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy

        acc_prob = min(1, np.exp(-(delta_energy+bias_energy)/self.kT))
        
        if np.random.rand() >= acc_prob:
            self.particles[particle_idx].position = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            # self.target_clust_idx = self.tmp_target_clust_idx
            self.rejected_rates[0] += 1

        self.energy_log.append(self.energy)

    # in -> out AVBMC move
    def inout_AVBMC(self, anchor_idx):
        self.attempts[1] += 1

        Nin, Nin_idx = self.calc_in(self.particles[anchor_idx])
        if Nin == 1 and 0 in Nin_idx:
            self.rejected_rates[1] += 1 
            return
        if Nin == 0:
            self.rejected_rates[1] += 1 
            return
        target_idx = np.random.choice(Nin_idx)
        while target_idx == 0:
            target_idx = np.random.choice(Nin_idx)

        old_energy = self.calc_energy(target_idx)
        old_pos = self.particles[target_idx].position

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
        new_pos[2] = np.round(np.random.rand(1) * (self.box_length-22))+11
        while self.calc_dist(old_pos, new_pos) <= self.clust_cutoff:
            new_pos = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
            new_pos[2] = np.round(np.random.rand(1) * (self.box_length-22))+11

        self.particles[target_idx].position = new_pos

        # self.tmp_target_clust_idx = self.target_clust_idx.copy()
        # self.target_clust_idx = self.find_cluster_around_target()
        # bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))
        bias_energy = 0.0    
        
        new_energy = self.calc_energy(target_idx)
        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy
        avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.kT)*self.Vout/self.Vin*(Nin)/(self.num_particles-Nin+1)

        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.particles[target_idx].position = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            # self.target_clust_idx = self.tmp_target_clust_idx
            self.rejected_rates[1] += 1

        self.energy_log.append(self.energy)

    # out -> in AVBMC move
    def outin_AVBMC(self, anchor_idx):
        self.attempts[2] += 1

        Nin, Nin_idx = self.calc_in(self.particles[anchor_idx])
        target_idx = np.random.randint(self.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0):
            target_idx = np.random.randint(self.num_particles)

        old_energy = self.calc_energy(target_idx)
        old_pos = self.particles[target_idx].position
        target_clust = self.find_cluster_around_target()

        # Calculate wnew for the new configuration
        nrb = 32  # Number of Rosenbluth trials
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.clust_cutoff * 2), 3)
            while ((sum(displacement**2) > self.clust_cutoff**2) or (sum(displacement**2) < self.low_cutoff**2)) or ((self.particles[anchor_idx].position[2] + displacement[2]) % self.box_length < 11) or ((self.particles[anchor_idx].position[2] + displacement[2]) % self.box_length > self.box_length-11):
                # while ((self.particles[anchor_idx].position[2] + displacement[2]) % self.box_length < 11) or ((self.particles[anchor_idx].position[2] + displacement[2]) % self.box_length > self.box_length-11):
                    displacement = np.round(((np.random.rand(3) - 0.5) * self.clust_cutoff * 2), 3)
            
            new_pos = (self.particles[anchor_idx].position + displacement) % (self.box_length)
            self.particles[target_idx].position = new_pos
            new_energy = self.calc_energy(target_idx)
            delta_energy = new_energy - old_energy
            w = np.exp(-delta_energy / self.kT)
            wnew += w
            rosenbluth_weights.append((w, new_energy, new_pos))
            self.particles[target_idx].position = old_pos

        # Select one configuration based on Rosenbluth weights
        rosenbluth_weights_norm = [(weight / wnew, d, pos) for weight, d, pos in rosenbluth_weights]
        # rosenbluth_weights_norm = [(weight / wnew, d, pos) if not np.isnan(weight / wnew) else (0, d, pos) for weight, d, pos in rosenbluth_weights]
        _, new_energy, selected_pos = rosenbluth_weights[np.random.choice(range(len(rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.particles[target_idx].position = selected_pos

        # Calculate wold for the original configuration
        wold = np.exp(-(old_energy - new_energy) / self.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            old_energy_out = new_energy
            
            old_pos_out = self.particles[target_idx_out].position
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
            new_pos_out[2] = np.round(np.random.rand(1) * (self.box_length-22))+11
            while self.calc_dist(old_pos_out, new_pos_out) <= self.clust_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.box_length * 2), 3) % self.box_length
                new_pos_out[2] = np.round(np.random.rand(1) * (self.box_length-22))+11

            self.particles[target_idx_out].position = new_pos_out
            new_energy_out = self.calc_energy(target_idx_out)
            delta_energy_out = new_energy_out - old_energy_out
            w = np.exp(-delta_energy_out / self.kT)
            wold += w
            self.particles[target_idx_out].position = old_pos_out

        # self.tmp_target_clust_idx = self.target_clust_idx.copy()
        # self.target_clust_idx = self.find_cluster_around_target()
        # bias_energy = self.bias.denergy(len(self.target_clust_idx), len(self.tmp_target_clust_idx))
        bias_energy = 0.0

        # bias_energy = self.bias.denergy(len(target_clust), len(target_clust)+1)
        delta_energy = new_energy - old_energy
        self.energy += delta_energy
        self.bias_energy += bias_energy

        avbmc_energy = np.exp(-bias_energy/self.kT) * (wnew/wold) * (self.vol_ratio) * ((self.num_particles - Nin) / (Nin + 1))
        # avbmc_energy = (wnew/wold) * (self.vol_ratio) * ((self.num_particles - Nin) / (Nin + 1))
        
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.particles[target_idx].position = old_pos
            self.energy -= delta_energy
            self.bias_energy -= bias_energy
            self.rejected_rates[2] += 1
            # self.target_clust_idx = self.tmp_target_clust_idx
        self.energy_log.append(self.energy)
    
