import numpy as np
import logging

logger = logging.getLogger('monte_carlo')

class Move:
    def __init__(self, system):
        '''Initialize base move class with system reference and statistics tracking'''
        self.system = system
        self.attempts = 0
        self.rejections = 0
    
    def get_acceptance_rate(self):
        '''Calculate and return current acceptance rate for this move type'''
        if self.attempts == 0:
            return 0.0
        return 1.0 - (self.rejections / self.attempts)
    
    def reset_stats(self):
        '''Reset move attempt and rejection counters to zero'''
        self.attempts = 0
        self.rejections = 0
    
    def attempt_move(self, particle_idx):
        '''Abstract method for attempting a Monte Carlo move - must be implemented by subclasses'''
        raise NotImplementedError("Subclasses must implement attempt_move")


class TranslationMove(Move):
    def __init__(self, system):
        '''Initialize translation move with maximum displacement parameter'''
        super().__init__(system)
        self.max_displacement = system.config.max_displacement

    def attempt_move(self, molecule_idx):
        '''Translate entire molecule (all atoms) by random displacement within spherical constraint'''
        self.attempts += 1

        # Generate random displacement within sphere
        displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)
        while (np.sum(displacement**2) > self.max_displacement**2):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)

        # Store old positions for all atoms in molecule
        atom_indices = self.system.molecules[molecule_idx]
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]
        old_com = self.system.get_molecule_com(molecule_idx)

        # Calculate new COM position with PBC
        new_com = (old_com + displacement) % self.system.box_length

        # Move all atoms by displacement (preserving molecular geometry)
        for idx in atom_indices:
            self.system.positions[idx] = (self.system.positions[idx] + displacement) % self.system.box_length

        # Calculate energy change
        if self.system.bias is None:
            old_energy = self.system.energy
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = 0.0
        else:
            old_cluster_size = len(self.system.find_target_cluster())
            old_energy = self.system.energy
            new_cluster_size = len(self.system.find_target_cluster())
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = self.system.bias.denergy(new_cluster_size, old_cluster_size)

        # Accept/reject with Metropolis criterion
        acc_prob = min(1, np.exp(np.clip((-(delta_energy + bias_energy) / self.system.kT), -500, 500)))
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept - update system energy
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class SwapMove(Move):
    def __init__(self, system):
        '''Initialize swap move for random molecular repositioning'''
        super().__init__(system)

    def attempt_move(self, molecule_idx):
        '''Relocate entire molecule to random position with random orientation'''
        from utils import random_rotation_matrix
        self.attempts += 1

        # Store old positions
        atom_indices = self.system.molecules[molecule_idx]
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]

        # Generate random COM position
        new_com = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        # For molecular species, also randomize orientation
        mol_type = self.system.molecule_type[molecule_idx]
        if len(atom_indices) > 1:  # Multi-atom molecule
            rotation = random_rotation_matrix()
            self.system.update_molecule_positions(molecule_idx, new_com, rotation)
        else:  # Single-atom molecule
            self.system.update_molecule_positions(molecule_idx, new_com)

        # Calculate energy change
        if self.system.bias is None:
            old_energy = self.system.energy
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = 0.0
        else:
            old_cluster_size = len(self.system.find_target_cluster())
            old_energy = self.system.energy
            new_cluster_size = len(self.system.find_target_cluster())
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = self.system.bias.denergy(new_cluster_size, old_cluster_size)

        # Accept/reject
        acc_prob = min(1, np.exp(np.clip((-(delta_energy + bias_energy) / self.system.kT), -500, 500)))
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class RotationMove(Move):
    def __init__(self, system):
        '''Initialize rotation move using Euler angles'''
        super().__init__(system)

    def attempt_move(self, molecule_idx):
        '''Rotate rigid molecule around its COM using random Euler angles'''
        from utils import random_rotation_matrix
        self.attempts += 1

        # Skip single-atom molecules (e.g., Ca)
        atom_indices = self.system.molecules[molecule_idx]
        if len(atom_indices) == 1:
            return

        # Store old positions
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]
        current_com = self.system.get_molecule_com(molecule_idx)

        # Generate random rotation using Euler angles (ZYZ convention)
        rotation = random_rotation_matrix()

        # Apply rotation around current COM
        self.system.update_molecule_positions(molecule_idx, current_com, rotation)

        # Calculate energy change
        if self.system.bias is None:
            old_energy = self.system.energy
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = 0.0
        else:
            old_cluster_size = len(self.system.find_target_cluster())
            old_energy = self.system.energy
            new_cluster_size = len(self.system.find_target_cluster())
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = self.system.bias.denergy(new_cluster_size, old_cluster_size)

        # Accept/reject
        acc_prob = min(1, np.exp(np.clip((-(delta_energy + bias_energy) / self.system.kT), -500, 500)))
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class InOutAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC in-out move with volume calculations for bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx):
        '''Move molecule from cluster to bulk with random position and orientation'''
        from utils import random_rotation_matrix
        self.attempts += 1

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)

        # Store old positions for all atoms in molecule
        atom_indices = self.system.molecules[target_idx]
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]
        old_com = self.system.get_molecule_com(target_idx)

        # Generate random position in bulk (far from old position)
        new_com = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
        while self.system.calc_dist(old_com, new_com) <= self.system.config.upper_cutoff:
            new_com = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        # For molecules, also randomize orientation
        mol_type = self.system.molecule_type[target_idx]
        if len(atom_indices) > 1:
            rotation = random_rotation_matrix()
            self.system.update_molecule_positions(target_idx, new_com, rotation)
        else:
            self.system.update_molecule_positions(target_idx, new_com)

        # Calculate energy change
        if self.system.bias is None:
            old_energy = self.system.energy
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = 0.0
        else:
            old_cluster_size = len(self.system.find_target_cluster())
            old_energy = self.system.energy
            new_cluster_size = len(self.system.find_target_cluster())
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = self.system.bias.denergy(new_cluster_size, old_cluster_size)

        avbmc_energy = np.exp(np.clip(-(delta_energy+bias_energy)/self.system.kT, -500, 500)) * self.Vout/self.Vin * (Nin)/(self.system.num_molecules-Nin+1)
        acc_prob = min(1, avbmc_energy)

        # Accept/reject
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class OutInAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC out-in move with Rosenbluth sampling over position and orientation'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx):
        '''Move molecule from bulk to cluster with Rosenbluth sampling over position+orientation'''
        from utils import random_rotation_matrix
        self.attempts += 1

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        target_idx = np.random.randint(self.system.num_molecules)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0):
            target_idx = np.random.randint(self.system.num_molecules)

        old_energy = self.system.energy
        atom_indices = self.system.molecules[target_idx]
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]
        self.system.target_clust_idx = self.system.find_target_cluster()

        # Rosenbluth sampling over position AND orientation
        nrb = 32
        new_energies = []
        new_configs = []  # Store (positions_list, rotation_matrix) tuples

        anchor_com = self.system.get_molecule_com(anchor_idx)

        for _ in range(nrb):
            # Sample position in spherical shell around anchor
            r = np.cbrt(np.random.rand() * (self.system.clust_cutoff**3 - self.system.config.lower_cutoff**3) + self.system.config.lower_cutoff**3)
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)

            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_com = (anchor_com + np.array([x, y, z])) % self.system.box_length

            # Sample random orientation for molecules
            if len(atom_indices) > 1:
                rotation = random_rotation_matrix()
                self.system.update_molecule_positions(target_idx, new_com, rotation)
            else:
                rotation = None
                self.system.update_molecule_positions(target_idx, new_com)

            trial_energy = self.system.calc_full_energy()
            new_energies.append(trial_energy)
            new_configs.append([self.system.positions[idx].copy() for idx in atom_indices])

            # Restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]

        # Calculate wold: old config + random bulk positions with random orientations
        old_energies = [old_energy]
        old_com = self.system.get_molecule_com(target_idx)

        for _ in range(nrb - 1):
            # Random position in bulk (far from old position)
            new_com_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_com, new_com_out) <= self.system.config.upper_cutoff:
                new_com_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

            # Random orientation for molecules
            if len(atom_indices) > 1:
                rotation = random_rotation_matrix()
                self.system.update_molecule_positions(target_idx, new_com_out, rotation)
            else:
                self.system.update_molecule_positions(target_idx, new_com_out)

            trial_energy = self.system.calc_full_energy()
            old_energies.append(trial_energy)

            # Restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]

        # Calculate Rosenbluth weights
        all_energies = new_energies + old_energies
        energy_ref = min(all_energies)

        wnew = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies)
        wold = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in old_energies)

        if wnew == 0:
            self.rejections += 1
            return

        # Select configuration based on Rosenbluth weights
        weights_new = [np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies]
        weights_norm = [w / wnew for w in weights_new]
        selected_idx = np.random.choice(len(new_energies), p=weights_norm)
        new_energy = new_energies[selected_idx]
        selected_config = new_configs[selected_idx]

        # Apply selected configuration
        for i, idx in enumerate(atom_indices):
            self.system.positions[idx] = selected_config[i]

        # Calculate bias energy
        if self.system.bias is not None:
            self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy

        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_molecules - Nin) / (Nin + 1))
        acc_prob = min(1, avbmc_energy)

        # Accept/reject
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy

class NVTInOutMove(Move):
    def __init__(self, system):
        '''Initialize NVT in-out move for cluster nucleation with AVBMC bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx, Nin_idx):
        '''Move molecule from target cluster to bulk with random position and orientation'''
        from utils import random_rotation_matrix
        self.attempts += 1
        Nin = len(Nin_idx)

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)

        # Store old positions
        atom_indices = self.system.molecules[target_idx]
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]

        # Ensure target cluster is current
        self.system.target_clust_idx = self.system.find_target_cluster()
        clust_coms = np.array([self.system.get_molecule_com(i) for i in self.system.target_clust_idx])

        # Generate random position far from all cluster molecules
        regen = True
        while regen:
            new_com = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            distances = np.linalg.norm(clust_coms - new_com, axis=1)
            if np.all(distances > self.system.config.upper_cutoff):
                regen = False

        # Randomize orientation for molecules
        if len(atom_indices) > 1:
            rotation = random_rotation_matrix()
            self.system.update_molecule_positions(target_idx, new_com, rotation)
        else:
            self.system.update_molecule_positions(target_idx, new_com)

        # Calculate energy change
        old_cluster_size = len(self.system.target_clust_idx)
        if self.system.bias is None:
            old_energy = self.system.energy
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = 0.0
        else:
            old_energy = self.system.energy
            new_cluster_size = len(self.system.find_target_cluster())
            new_energy = self.system.calc_full_energy()
            delta_energy = new_energy - old_energy
            bias_energy = self.system.bias.denergy(new_cluster_size, old_cluster_size)

        try:
            avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.system.kT)*self.Vout/self.Vin*(Nin)/(self.system.num_molecules-old_cluster_size+1)*(old_cluster_size/(old_cluster_size-1))
        except:
            avbmc_energy = 0
        acc_prob = min(1, avbmc_energy)

        # Accept/reject
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class NVTOutInMove(Move):
    def __init__(self, system):
        '''Initialize NVT out-in move with Rosenbluth sampling over position and orientation'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx, Nin_idx):
        '''Move molecule from bulk to target cluster with Rosenbluth sampling over position+orientation'''
        from utils import random_rotation_matrix
        self.attempts += 1
        Nin = len(Nin_idx)

        target_idx = np.random.randint(self.system.num_molecules)
        self.system.target_clust_idx = self.system.find_target_cluster()
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0) or (target_idx in self.system.target_clust_idx):
            target_idx = np.random.randint(self.system.num_molecules)

        old_energy = self.system.energy
        atom_indices = self.system.molecules[target_idx]
        old_positions = [self.system.positions[idx].copy() for idx in atom_indices]
        self.system.target_clust_idx = self.system.find_target_cluster()

        # Rosenbluth sampling over position AND orientation
        nrb = 32
        new_energies = []
        new_configs = []

        anchor_com = self.system.get_molecule_com(anchor_idx)

        for _ in range(nrb):
            # Sample position in spherical shell
            displacement = np.round(((np.random.rand(3) - 0.5) * self.system.config.upper_cutoff * 2), 3)
            while ((sum(displacement**2) > self.system.config.upper_cutoff**2) or (sum(displacement**2) < self.system.config.lower_cutoff**2)):
                displacement = np.round(((np.random.rand(3) - 0.5) * self.system.config.upper_cutoff * 2), 3)

            new_com = (anchor_com + displacement) % self.system.box_length

            # Sample random orientation
            if len(atom_indices) > 1:
                rotation = random_rotation_matrix()
                self.system.update_molecule_positions(target_idx, new_com, rotation)
            else:
                self.system.update_molecule_positions(target_idx, new_com)

            trial_energy = self.system.calc_full_energy()
            new_energies.append(trial_energy)
            new_configs.append([self.system.positions[idx].copy() for idx in atom_indices])

            # Restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]

        # Calculate wold: old config + random bulk positions with random orientations
        old_energies = [old_energy]
        old_com = self.system.get_molecule_com(target_idx)

        for _ in range(nrb - 1):
            # Random position in bulk
            new_com_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_com, new_com_out) <= self.system.config.upper_cutoff:
                new_com_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

            # Random orientation
            if len(atom_indices) > 1:
                rotation = random_rotation_matrix()
                self.system.update_molecule_positions(target_idx, new_com_out, rotation)
            else:
                self.system.update_molecule_positions(target_idx, new_com_out)

            trial_energy = self.system.calc_full_energy()
            old_energies.append(trial_energy)

            # Restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]

        # Calculate Rosenbluth weights with reference energy for numerical stability
        all_energies = new_energies + old_energies
        energy_ref = min(all_energies)

        wnew = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies)
        wold = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in old_energies)

        if wnew == 0:
            self.rejections += 1
            return

        # Select configuration based on Rosenbluth weights
        weights_new = [np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies]
        weights_norm = [w / wnew for w in weights_new]
        selected_idx = np.random.choice(len(new_energies), p=weights_norm)
        new_energy = new_energies[selected_idx]
        selected_config = new_configs[selected_idx]

        # Apply selected configuration
        for i, idx in enumerate(atom_indices):
            self.system.positions[idx] = selected_config[i]

        # Calculate bias energy
        self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
        if self.system.bias is not None:
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy

        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_molecules - len(self.system.tmp_target_clust_idx)) / (Nin + 1)) * ((len(self.system.tmp_target_clust_idx)) / (len(self.system.tmp_target_clust_idx)+1))

        acc_prob = min(1, avbmc_energy)

        # Accept/reject
        if np.random.rand() >= acc_prob:
            # Reject - restore old positions
            for i, idx in enumerate(atom_indices):
                self.system.positions[idx] = old_positions[i]
            self.rejections += 1
        else:
            # Accept
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy
