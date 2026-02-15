import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from collections import deque

from utils import *
import logging

class System:
    def __init__(self, config, id=0):
        '''Initialize System with configuration parameters, PMF, and move objects'''
        self.config = config
        self.box_length = config.box_length
        self.num_particles = config.num_particles
        self.kT = config.kT

        self.id = str(id).zfill(2)
        self.pmf = PMF("potentials/"+config.ff_path)
        self.seed = config.seed + id
        np.random.seed(self.seed)
        
        # Initialize atomic properties
        self.positions = []
        self.types = []
        self.target_clust_idx = []
        self.cluster_sizes = []
        self.clust_cutoff = config.clust_cutoff
        
        # Initialize move objects dynamically from config
        self.active_moves = []
        self.move_names = []
        self.move_probabilities = config.move_probabilities
        for move_name, rate, move_class in config.active_moves:
            move_instance = move_class(self)
            self.active_moves.append(move_instance)
            self.move_names.append(move_name)

        # initialize bias as specified in config file
        self.bias = None
        if config.bias_type is not None:
            self.bias = config.bias

    def init_positions(self, input_path=None, multi=False):
        '''Initialize particle positions either from input file or randomly, ensuring proper ratios and minimum separations'''
        if input_path:
            filename = input_path.strip(".xyz") + f"_{self.id}.xyz" if multi else input_path
            with open(f"inputs/{filename}", 'r') as f:
                lines = f.readlines()

            data_lines = lines[2:]
            part_types = {line.split()[0] for line in data_lines}
            num_parts = len(data_lines)

            if num_parts != self.num_particles:
                raise ValueError(f"{filename} has {num_parts} particles, expected {self.num_particles}")
            if len(part_types) != 2:
                raise ValueError(f"{filename} must contain exactly two particle types, found {len(part_types)}: {part_types}")

            part_types = list(part_types)
            assign_value = lambda atom: 0 if atom == part_types[0] else 1

            for line in data_lines:
                atom, x, y, z = line.split()
                self.positions.append(np.array([float(x), float(y), float(z)]))
                self.types.append(assign_value(atom))

            type_counts = [self.types.count(0), self.types.count(1)]
            expected = lambda r: (self.num_particles // self.config.total_ratio) * r
            expected_type1 = expected(self.config.ratio_type1)
            expected_type2 = expected(self.config.ratio_type2)

            if type_counts != [expected_type1, expected_type2]:
                print(f"WARNING: Input ratio {type_counts[0]}:{type_counts[1]} ≠ expected {expected_type1}:{expected_type2}")

        else:
            # Calculate number of each ion type based on ratio
            formula_units = self.num_particles // self.config.total_ratio
            n_type1 = formula_units * self.config.ratio_type1
            n_type2 = formula_units * self.config.ratio_type2
            
            is_valid = lambda pos, existing: all(
                np.linalg.norm(
                    np.mod(pos - p + 0.5 * self.box_length, self.box_length) - 0.5 * self.box_length
                ) >= self.config.lower_energy_cutoff
                for p in existing
            )

            # populate first ion type
            for _ in range(n_type1):
                tries = 0
                while True:
                    position = np.round(np.random.rand(3) * self.box_length, 3)
                    if is_valid(position, self.positions):
                        self.positions.append(position)
                        self.types.append(0)
                        break
                    tries += 1
                    if tries >= 1000:
                        logging.warning(f"Could not place type 0 particle after 1000 tries. Placing anyway (may overlap).")
                        logging.warning(f"Energy may jump significantly due to overlap.")
                        self.positions.append(position)
                        self.types.append(0)
                        break

            # populate second ion type
            for _ in range(n_type2):
                tries = 0
                while True:
                    position = np.round(np.random.rand(3) * self.box_length, 3)
                    if is_valid(position, self.positions):
                        self.positions.append(position)
                        self.types.append(1)
                        break
                    tries += 1
                    if tries >= 1000:
                        logging.warning(f"Could not place type 1 particle after 1000 tries. Placing anyway (may overlap).")
                        logging.warning(f"Energy may jump significantly due to overlap.")
                        self.positions.append(position)
                        self.types.append(1)
                        break
        
        self.positions = np.array(self.positions)
        self.types = np.array(self.types)
        
        self.target_clust_idx = self.find_target_cluster()
        self.calc_full_energy()

    def step(self):
        '''Execute one Monte Carlo step by randomly selecting and attempting a move'''
        # iterate over all particles in system
        for particle in range(self.num_particles):
            # Dynamic move selection using probabilities from configq
            move_idx = np.random.choice(len(self.active_moves), p=self.move_probabilities)

            # Update target cluster, if particle is not in it, skip NVT move
            if 'nvt' in self.move_names[move_idx]:
                self.target_clust_idx = self.find_target_cluster()
            while particle not in self.target_clust_idx and 'nvt' in self.move_names[move_idx]:
                move_idx = np.random.choice(len(self.active_moves), p=self.move_probabilities)
            
            selected_move = self.active_moves[move_idx]
            move_name = self.move_names[move_idx]

            # NVT moves need special handling
            if 'nvt' in move_name or 'avbmc' in move_name:
                # pick random particle type to calculate Nin
                part_type = np.random.choice(np.unique(self.types))
                Nin, Nin_idx = self.calc_in(particle, part_type=part_type)
                selected_move.attempt_move(particle, Nin_idx, part_type)
            else:
                selected_move.attempt_move(particle)

    def calc_energy_delta(self, particle_idx, new_pos, old_pos):
        '''Calculate energy difference between new and old positions, including bias energy if applicable'''
        if self.bias is None:
            self.positions[particle_idx] = old_pos
            old_energy = self.calc_energy(particle_idx)
            self.positions[particle_idx] = new_pos
            new_energy = self.calc_energy(particle_idx)
            delta_energy = new_energy - old_energy
            delta_bias_energy = 0.0
        # If bias is active must calculate change in cluster size
        else:
            old_cluster = self.find_target_cluster()
            old_cluster_len = len(old_cluster)
            self.positions[particle_idx] = old_pos
            old_energy = self.calc_energy(particle_idx)
            self.positions[particle_idx] = new_pos
            new_cluster = self.find_target_cluster()
            new_cluster_len = len(new_cluster)
            new_energy = self.calc_energy(particle_idx)
            delta_energy = new_energy - old_energy
            delta_bias_energy = self.bias.denergy(new_cluster_len, old_cluster_len)

        self.positions[particle_idx] = old_pos  # Reset position after calculation
        return delta_energy, delta_bias_energy

    def calc_energy(self, particle_idx):
        '''Calculate total energy of a particle with all other particles using tabulated PMF'''
        return calc_energy_numba(
            self.positions, self.types, particle_idx, 
            self.config.lower_energy_cutoff, self.config.energy_cutoff,
            self.box_length, self.pmf.sorted_distances,
            self.pmf.energy_columns[0], self.pmf.energy_columns[1], self.pmf.energy_columns[2]
        )
    
    def calc_full_energy(self):
        '''Calculate total system energy including bias contribution if applicable'''
        # Calculate total energy of the system
        self.energy = 0.0
        for i in range(self.num_particles):
            self.energy += self.calc_energy(i)
        self.energy /= 2.0  # Each interaction counted twice

        # Set initial bias energy if applicable
        if self.bias is not None:
            self.bias_energy = self.bias.energy(len(self.target_clust_idx))
        else:
            self.bias_energy = 0.0
        return self.energy
    
    def find_target_cluster(self, target_idx=0):
        '''Find all particles connected to target particle within cluster cutoff distance using breadth-first search'''        
        # Start with the target particle
        # tree = cKDTree(self.positions, boxsize=self.box_length+0.001)
        visited = set()
        queue = [target_idx]
        cluster = []
        
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                cluster.append(current)
                # neighbors = tree.query_ball_point(self.positions[current], self.clust_cutoff)
                # for neighbor in neighbors:
                    # if neighbor not in visited:
                        # queue.append(neighbor)
                neighbors = find_neighbors_numba(self.positions, self.positions[current], self.clust_cutoff, self.box_length)
                # check if nieghbors are different type from target particle
                neighbors = [n for n in neighbors if self.types[n] != self.types[current]]
                queue.extend([n for n in neighbors if n not in visited])
        
        # Cluster around target particle found
        # self.target_clust_idx = cluster
        return cluster
    
    def find_clusters(self):
        '''Find all clusters in the system using Stillinger cluster analysis with periodic boundary conditions'''
        tree = cKDTree(self.positions, boxsize=self.box_length + 1e-6)  # slight offset avoids precision issues
        neighbor_lists = tree.query_ball_point(self.positions, self.clust_cutoff)

        visited = np.zeros(self.num_particles, dtype=bool)
        clusters = []

        for i in range(self.num_particles):
            if not visited[i]:
                cluster = []
                queue = deque([i])
                while queue:
                    current = queue.popleft()
                    if not visited[current]:
                        visited[current] = True
                        cluster.append(current)
                        for neighbor in neighbor_lists[current]:
                            # if not visited[neighbor]:
                            if not visited[neighbor] and self.types[neighbor] != self.types[current]:
                                queue.append(neighbor)
                clusters.append(cluster)

        self.cluster_sizes = [len(c) for c in clusters]
        self.target_clust_idx = clusters[0]
        return self.cluster_sizes, self.target_clust_idx
        
    def check_in(self, particle_idx):
        '''Check if particle is within cluster cutoff distance of any particle in target cluster'''
        for clust_idx in self.target_clust_idx:
            if clust_idx != particle_idx:
                distance = self.calc_dist(self.positions[particle_idx], self.positions[clust_idx])
                if distance < self.clust_cutoff:
                    return True
        return False
        
    def calc_in(self, particle_idx, part_type=None):
        '''Calculate neighbors within cluster cutoff distance using PBC distances'''
        pos = self.positions[particle_idx]
        pos_diff = self.positions - pos
        pos_diff = pos_diff - self.box_length * np.round(pos_diff / self.box_length)
        distances = np.linalg.norm(pos_diff, axis=1)
        
        # Find neighbors within cutoff (excluding the particle itself)
        within_cutoff = (distances < self.clust_cutoff) & (distances > 0.0)
        neighbors = np.where(within_cutoff)[0].tolist()
        # select only neighbors of type that is specified
        if part_type is not None:
            neighbors = [n for n in neighbors if self.types[n] != part_type]
        
        Nin = len(neighbors)
        Nin_idx = neighbors
        
        return Nin, Nin_idx

    
    def calc_dist(self, pos1, pos2):
        '''Calculate minimum image distance between two positions with periodic boundary conditions'''
        dist_vec = np.abs(pos1 - pos2)
        dist_vec = dist_vec - self.box_length * np.round(dist_vec / self.box_length)
        return np.linalg.norm(dist_vec)
