import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from collections import deque

from utils import *
import logging

class System:
    def __init__(self, config, id=0, debug=False):
        '''Initialize System with configuration parameters, PMF, and move objects'''
        self.config = config
        self.box_length = config.box_length
        self.num_particles = config.num_particles
        self.kT = config.kT
        self.energy = 0.0
        self.bias_energy = 0.0

        # Debug logging: when on, each move stashes a record here for the runner to write
        self.debug = debug
        self._debug = None

        self.id = str(id).zfill(2)
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

    def init_pmf(self):
        '''Initialize PMF object with potential file from config'''
        charge_map = {0: 1.0, 1: -1.0}  # Example charge mapping for two types
        self.charges = [charge_map[t] for t in self.types]
        self.pmf = PMF(self.types, self.charges, self.box_length, self.config.model_path)

    def init_positions(self, input_path=None, multi=False):
        '''Initialize particle positions either from input file or randomly, ensuring proper ratios and minimum separations'''
        if input_path:
            # used for US simulations where we want each markov chain to start from a different seed
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
        self.init_pmf()
        self.energy = 0.0  # Will be calculated in equilibration/production run

    def step(self, step_num=0):
        '''Execute one Monte Carlo step by randomly selecting and attempting a move'''
        # Dynamic move selection using probabilities from config
        if not self.active_moves:
            return  # No moves configured

        self._debug = None  # cleared each step so stale records aren't re-logged

        move_idx = np.random.choice(len(self.active_moves),
                                   p=self.move_probabilities)
        selected_move = self.active_moves[move_idx]
        move_name = self.move_names[move_idx]

        particle = np.random.randint(self.config.num_particles)

        # Track which move instance actually runs, and its rejection count beforehand, so
        # debug logging can infer acceptance without touching any move's accept/reject code.
        if 'nvt' in move_name:
            self.find_target_cluster()
            particle = np.random.choice(self.target_clust_idx)
            Nin, Nin_idx = self.calc_in(particle)
            invoked = selected_move
            rej_before = invoked.rejections
            invoked.attempt_move(particle, Nin_idx)

        elif move_name == 'inout_avbmc':
            Nin, Nin_idx = self.calc_in(particle)
            if Nin >= 1:
                invoked = selected_move
            else:
                invoked = next(self.active_moves[mi] for mi, mn in enumerate(self.move_names)
                               if mn == 'outin_avbmc')
            rej_before = invoked.rejections
            invoked.attempt_move(particle)
        else:
            invoked = selected_move
            rej_before = invoked.rejections
            invoked.attempt_move(particle)

        if self.debug and self._debug is not None:
            self._debug += (1 if invoked.rejections == rej_before else 0,)

    def calc_energy_delta(self, particle_idx, new_pos, old_pos):
        '''Calculate energy difference between new and old positions, including bias energy if applicable'''
        # Snapshot the cluster so a rejected move can restore it (see restore_cluster).
        self._clust_before = list(self.target_clust_idx)
        if self.bias is None:
            old_energy = self.energy
            self.positions[particle_idx] = new_pos
            new_energy = self.calc_full_energy()
            delta_energy = new_energy - old_energy
            delta_bias_energy = 0.0
        # If bias is active must calculate change in cluster size
        else:
            old_cluster = len(self.target_clust_idx)
            old_energy = self.energy
            self.positions[particle_idx] = new_pos

            # Always recompute the target cluster fully (updates cache).
            new_cluster = len(self.find_target_cluster())

            new_energy = self.calc_full_energy()
            delta_energy = new_energy - old_energy
            delta_bias_energy = self.bias.denergy(new_cluster, old_cluster)

            if self.debug:
                self._debug = (old_energy, new_energy, old_cluster, new_cluster, delta_bias_energy)

        self.positions[particle_idx] = old_pos  # Reset position after calculation
        return delta_energy, delta_bias_energy

    def restore_cluster(self):
        '''Restore the target cluster cache (index list and size) after a rejected move.'''
        self.target_clust_idx = self._clust_before

    def calc_full_energy(self):
        '''Calculate total energy of the system using physics prior and MLP'''
        return self.pmf.energies(self.positions)

    def find_target_cluster(self, target_idx=0):
        '''Find all particles connected to target particle within cluster cutoff distance using breadth-first search'''
        visited = set()
        queue = deque([target_idx])
        cluster = []

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                cluster.append(current)
                neighbors = find_neighbors_numba(self.positions, self.positions[current], self.clust_cutoff, self.box_length)
                queue.extend(n for n in neighbors if n not in visited)

        self.target_clust_idx = cluster
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
                            if not visited[neighbor]:
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
        
    def calc_in(self, particle_idx):
        '''Calculate neighbors within cluster cutoff distance using PBC distances.'''
        # Calculate distances from particle to all others with PBC
        pos = self.positions[particle_idx]
        pos_diff = self.positions - pos
        pos_diff = pos_diff - self.box_length * np.round(pos_diff / self.box_length)
        distances = np.linalg.norm(pos_diff, axis=1)
        
        # Find neighbors within cutoff (excluding the particle itself)
        within_cutoff = (distances < self.clust_cutoff) & (distances > 0.0)
        neighbors = np.where(within_cutoff)[0].tolist()
        
        Nin = len(neighbors)
        Nin_idx = neighbors
        
        return Nin, Nin_idx
    
    def calc_dist(self, pos1, pos2):
        '''Calculate minimum image distance between two positions with periodic boundary conditions'''
        dist_vec = np.abs(pos1 - pos2)
        dist_vec = dist_vec - self.box_length * np.round(dist_vec / self.box_length)
        return np.linalg.norm(dist_vec)

    def unwrap_positions(self, cluster_indices):
        '''Unwrap cluster particle positions relative to the first particle, removing periodic boundary jumps'''
        if len(cluster_indices) == 0:
            return np.array([])
        reference_pos = self.positions[cluster_indices[0]]
        unwrapped = np.zeros((len(cluster_indices), 3))
        unwrapped[0] = reference_pos
        for i, idx in enumerate(cluster_indices[1:], start=1):
            delta = self.positions[idx] - reference_pos
            delta -= self.box_length * np.round(delta / self.box_length)
            unwrapped[i] = reference_pos + delta
        return unwrapped

    def calc_rcut(self, clust=None, coordinates=False):
        '''Calculate rcut as the maximum distance from the geometric center of the target cluster to its particles'''
        clust = self.find_target_cluster() if clust is None else clust
        positions = self.unwrap_positions(clust)
        geometric_center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - geometric_center, axis=1)
        rcut = np.max(distances)
        if coordinates:
            translated_positions = positions - geometric_center + self.box_length / 2
            return rcut, translated_positions, self.types[clust]
        return rcut
