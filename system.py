import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from collections import deque
import logging

from utils import PMFLammps

class System:
    def __init__(self, config, id=0):
        '''Initialize System with configuration parameters, PMF, and move objects'''
        self.config = config
        self.box_length = config.box_length
        self.num_particles = config.num_particles
        self.kT = config.kT
        self.energy = 0.0
        self.bias_energy = 0.0

        self.id = str(id).zfill(2)
        self.seed = config.seed + id
        np.random.seed(self.seed)

        # Initialize atomic properties
        self.positions = []
        self.types = []
        self.target_clust_idx = []
        self.cluster_sizes = []
        self.clust_cutoff = config.clust_cutoff

        # Initialize molecular properties
        self.molecules = []  # List of tuples: each tuple contains atom indices for that molecule
        self.molecule_type = []  # Array of molecule types (0=Ca, 1=CO3, etc.)
        self.num_molecules = 0

        # Load rigid molecular geometries from .body files (body frame coordinates)
        # Molecule type 0: Ca (single atom, no geometry needed)
        # Molecule type 1: CO3 (loaded from file)
        self.molecular_geometries = {}
        if hasattr(config, 'body_file') and config.body_file:
            self.molecular_geometries[1] = self._load_body_file(config.body_file)
        else:
            # Default CO3 geometry if no file specified
            self.molecular_geometries[1] = np.array([
                [0.0, 0.0, 0.0],
                [1.29, 0.0, 0.0],
                [-0.645, 1.117, 0.0],
                [-0.645, -1.117, 0.0]
            ])

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
    
    def _load_body_file(self, filepath):
        '''Load reference body frame geometry from .body file (x, y, z coordinates in Angstroms)'''
        positions = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        positions.append([x, y, z])
        return np.array(positions)

    def init_pmf(self):
        '''Initialize energy calculator (LAMMPS or debug soft-sphere potential)'''
        if hasattr(self.config, 'debug_potential') and self.config.debug_potential:
            from debug_potentials import SoftSpherePotential
            print("Using debug")
            self.pmf = SoftSpherePotential(epsilon=1.0, sigma=2.0, cutoff=self.config.upper_cutoff)
            logging.info("Using debug soft-sphere potential")
        else:
            self.pmf = PMFLammps(self.types, self.box_length, self.config.model_path, self.config.table_path)

    def init_positions(self, input_path=None, multi=False):
        '''Build molecular system with Ca atoms (1 atom each) and rigid CO3 molecules (4 atoms each)'''
        from utils import random_rotation_matrix

        if input_path:
            raise NotImplementedError("Molecular input files not yet implemented")

        else:
            # For CaCO3: num_particles is total ATOMS, ratio is Ca:CO3 MOLECULES
            # Each CaCO3 formula unit = 1 Ca (1 atom) + 1 CO3 (4 atoms) = 5 atoms total
            # For ratio 1:1, we have equal number of Ca and CO3 molecules

            atoms_per_formula_unit = 1 * self.config.ratio_type1 + 4 * self.config.ratio_type2  # Ca atoms + CO3 atoms
            formula_units = self.num_particles // atoms_per_formula_unit

            n_ca = formula_units * self.config.ratio_type1   # Number of Ca molecules
            n_co3 = formula_units * self.config.ratio_type2  # Number of CO3 molecules

            # Verify total atoms matches
            expected_atoms = n_ca + 4 * n_co3
            if expected_atoms != self.num_particles:
                raise ValueError(f"num_particles={self.num_particles} must be multiple of {atoms_per_formula_unit}. Got {n_ca} Ca + {n_co3} CO3 = {expected_atoms} atoms")

            # Overlap checking: ensure molecular COMs are separated
            is_valid_com = lambda com, existing_coms: all(
                self.calc_dist(com, existing_com) >= self.config.lower_energy_cutoff
                for existing_com in existing_coms
            )

            com_positions = []  # Track all molecular COMs for overlap checking

            # Place Ca²⁺ ions (single-atom molecules)
            for _ in range(n_ca):
                tries = 0
                while True:
                    com = np.round(np.random.rand(3) * self.box_length, 3)
                    if is_valid_com(com, com_positions):
                        atom_idx = len(self.positions)
                        self.positions.append(com)
                        self.types.append(0)  # Atom type: Ca
                        self.molecules.append((atom_idx,))  # Single-atom molecule
                        self.molecule_type.append(0)  # Molecule type: Ca
                        com_positions.append(com)
                        break
                    tries += 1
                    if tries >= 1000:
                        logging.warning(f"Could not place Ca after 1000 tries. Placing anyway (may overlap).")
                        atom_idx = len(self.positions)
                        self.positions.append(com)
                        self.types.append(0)
                        self.molecules.append((atom_idx,))
                        self.molecule_type.append(0)
                        com_positions.append(com)
                        break

            # Place CO3²⁻ molecules (4 atoms each: C + 3 O)
            for _ in range(n_co3):
                tries = 0
                while True:
                    com = np.round(np.random.rand(3) * self.box_length, 3)
                    if is_valid_com(com, com_positions):
                        # Generate random orientation
                        rotation = random_rotation_matrix()

                        # Get reference geometry and apply rotation
                        ref_geom = self.molecular_geometries[1].copy()  # CO3 geometry
                        rotated_geom = ref_geom @ rotation.T

                        # Place all atoms in molecule
                        atom_indices = []
                        for i, local_pos in enumerate(rotated_geom):
                            atom_pos = (com + local_pos) % self.box_length
                            atom_idx = len(self.positions)
                            self.positions.append(atom_pos)
                            if i == 0:
                                self.types.append(1)  # C atom
                            else:
                                self.types.append(2)  # O atom
                            atom_indices.append(atom_idx)

                        self.molecules.append(tuple(atom_indices))
                        self.molecule_type.append(1)  # Molecule type: CO3
                        com_positions.append(com)
                        break
                    tries += 1
                    if tries >= 1000:
                        logging.warning(f"Could not place CO3 after 1000 tries. Placing anyway (may overlap).")
                        rotation = random_rotation_matrix()
                        ref_geom = self.molecular_geometries[1].copy()
                        rotated_geom = ref_geom @ rotation.T
                        atom_indices = []
                        for i, local_pos in enumerate(rotated_geom):
                            atom_pos = (com + local_pos) % self.box_length
                            atom_idx = len(self.positions)
                            self.positions.append(atom_pos)
                            self.types.append(1 if i == 0 else 2)
                            atom_indices.append(atom_idx)
                        self.molecules.append(tuple(atom_indices))
                        self.molecule_type.append(1)
                        com_positions.append(com)
                        break

        self.positions = np.array(self.positions)
        self.types = np.array(self.types)
        self.molecule_type = np.array(self.molecule_type)
        self.num_molecules = len(self.molecules)

        self.target_clust_idx = self.find_target_cluster()
        self.init_pmf()
        self.energy = 0.0  # Will be calculated in equilibration/production run

    def get_molecule_com(self, molecule_idx):
        '''Calculate molecular COM from atomic positions (uniform mass weighting)'''
        atom_indices = self.molecules[molecule_idx]
        atom_positions = self.positions[list(atom_indices)]
        com = np.mean(atom_positions, axis=0)
        return com

    def update_molecule_positions(self, molecule_idx, new_com, rotation_matrix=None):
        '''Update all atom positions in molecule: translate to new_com and optionally rotate around COM'''
        atom_indices = self.molecules[molecule_idx]
        mol_type = self.molecule_type[molecule_idx]

        # Single-atom molecule (e.g., Ca) - just set position
        if len(atom_indices) == 1:
            self.positions[atom_indices[0]] = new_com

        # Multi-atom molecule - apply rotation and translation
        else:
            ref_geometry = self.molecular_geometries[mol_type].copy()

            # Apply rotation if provided
            if rotation_matrix is not None:
                rotated_geometry = ref_geometry @ rotation_matrix.T
            else:
                rotated_geometry = ref_geometry

            # Translate to new COM
            for i, atom_idx in enumerate(atom_indices):
                self.positions[atom_idx] = new_com + rotated_geometry[i]

    def calc_molecule_dist(self, mol_idx1, mol_idx2):
        '''Calculate PBC distance between two molecular COMs'''
        com1 = self.get_molecule_com(mol_idx1)
        com2 = self.get_molecule_com(mol_idx2)
        return self.calc_dist(com1, com2)

    def step(self, step_num=0):
        '''Execute one MC step: select random molecule and attempt a move based on probabilities'''
        if not self.active_moves:
            return

        energy_before = self.energy

        move_idx = np.random.choice(len(self.active_moves), p=self.move_probabilities)
        selected_move = self.active_moves[move_idx]
        move_name = self.move_names[move_idx]

        # Select random molecule (not atom)
        molecule = np.random.randint(self.num_molecules)

        if 'nvt' in move_name:
            # NVT moves select from target cluster
            self.find_target_cluster()
            molecule = np.random.choice(self.target_clust_idx)
            Nin, Nin_idx = self.calc_in(molecule)
            selected_move.attempt_move(molecule, Nin_idx)

        elif move_name == 'inout_avbmc':
            # Check if inout move is possible, fallback to outin if not
            Nin, Nin_idx = self.calc_in(molecule)
            if Nin >= 1:
                selected_move.attempt_move(molecule)
            else:
                for idx, name in enumerate(self.move_names):
                    if name == 'outin_avbmc':
                        self.active_moves[idx].attempt_move(molecule)
                        break
        else:
            selected_move.attempt_move(molecule)

    def calc_full_energy(self):
        '''Calculate total system energy using active energy calculator'''
        if hasattr(self.pmf, 'energy'):
            # Debug potential
            return self.pmf.energy(self.positions, self.types, self.box_length)
        else:
            # LAMMPS
            return self.pmf.energies(self.positions)

    def find_target_cluster(self, target_idx=0):
        '''Find all molecules connected to target molecule within cluster cutoff using BFS on molecular COMs'''
        visited = set()
        queue = [target_idx]
        cluster = []

        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                cluster.append(current)

                # Find neighboring molecules based on COM distances
                current_com = self.get_molecule_com(current)
                for mol_idx in range(self.num_molecules):
                    if mol_idx not in visited:
                        neighbor_com = self.get_molecule_com(mol_idx)
                        dist = self.calc_dist(current_com, neighbor_com)
                        if dist < self.clust_cutoff:
                            queue.append(mol_idx)

        return cluster
    
    def find_clusters(self):
        '''Find all molecular clusters using Stillinger analysis on molecular COM distances with PBC'''
        visited = np.zeros(self.num_molecules, dtype=bool)
        clusters = []

        # Build molecular COM positions for efficient neighbor search
        mol_coms = np.array([self.get_molecule_com(i) for i in range(self.num_molecules)])
        tree = cKDTree(mol_coms, boxsize=self.box_length + 1e-6)
        neighbor_lists = tree.query_ball_point(mol_coms, self.clust_cutoff)

        for i in range(self.num_molecules):
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
        
    def check_in(self, molecule_idx):
        '''Check if molecule COM is within cluster cutoff of any molecule in target cluster'''
        mol_com = self.get_molecule_com(molecule_idx)
        for clust_idx in self.target_clust_idx:
            if clust_idx != molecule_idx:
                clust_com = self.get_molecule_com(clust_idx)
                distance = self.calc_dist(mol_com, clust_com)
                if distance < self.clust_cutoff:
                    return True
        return False

    def calc_in(self, molecule_idx):
        '''Calculate neighboring molecules within cluster cutoff distance using COM-COM PBC distances'''
        mol_com = self.get_molecule_com(molecule_idx)

        # Calculate COM positions for all molecules
        all_coms = np.array([self.get_molecule_com(i) for i in range(self.num_molecules)])

        # Calculate distances with PBC
        pos_diff = all_coms - mol_com
        pos_diff = pos_diff - self.box_length * np.round(pos_diff / self.box_length)
        distances = np.linalg.norm(pos_diff, axis=1)

        # Find neighbors within cutoff (excluding the molecule itself)
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
