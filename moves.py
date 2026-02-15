import numpy as np

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
    
    def attempt_move(self, particle_idx):
        '''Attempt random translation move within spherical displacement constraint'''
        self.attempts += 1

        displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)
        while (np.sum(displacement**2) > self.max_displacement**2):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.max_displacement * 2), 3)

        old_pos = self.system.positions[particle_idx].copy()
        new_pos = (old_pos + displacement) % (self.system.box_length)

        delta_energy, bias_energy = self.system.calc_energy_delta(particle_idx, new_pos, old_pos)
        acc_prob = min(1, np.exp(np.clip((-(delta_energy+bias_energy)/self.system.kT), -500, 500)))
        if np.random.rand() >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[particle_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class SwapMove(Move):
    def __init__(self, system):
        '''Initialize swap move for random particle repositioning'''
        super().__init__(system)
    
    def attempt_move(self, particle_idx):
        '''Attempt to swap particle to random position in simulation box'''
        self.attempts += 1

        old_pos = self.system.positions[particle_idx].copy()
        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        delta_energy, bias_energy = self.system.calc_energy_delta(particle_idx, new_pos, old_pos)
        
        acc_prob = min(1, np.exp(np.clip((-(delta_energy+bias_energy)/self.system.kT), -500, 500)))
        if np.random.rand() >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[particle_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class InOutAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC in-out move with volume calculations for bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.config.upper_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx, Nin_idx, part_type):
        '''Attempt AVBMC move to remove particle from cluster to bulk solution'''
        self.attempts += 1

        Nin = len(Nin_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)
        old_pos = self.system.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
        while self.system.calc_dist(old_pos, new_pos) <= self.system.config.upper_cutoff:
            new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        delta_energy, bias_energy = self.system.calc_energy_delta(target_idx, new_pos, old_pos)

        Nout = len([i for i in range(self.system.num_particles) if self.system.types[i] == part_type]) - Nin
        avbmc_energy = np.exp(np.clip(-(delta_energy+bias_energy)/self.system.kT,
                                -500, 500)) * self.Vout/self.Vin * (Nin)/(Nout+1)
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[target_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class OutInAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC out-in move with volume calculations and Rosenbluth sampling'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.config.upper_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3
    
    def attempt_move(self, anchor_idx, Nin_idx, part_type):
        '''Attempt AVBMC move to insert bulk particle into cluster using Rosenbluth weighting'''
        self.attempts += 1

        # Nin, Nin_idx = self.system.calc_in(anchor_idx, opp_type=True)
        Nin = len(Nin_idx)
        target_idx = np.random.randint(self.system.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0) or (self.system.types[target_idx] != part_type):
            target_idx = np.random.randint(self.system.num_particles)

        old_energy = self.system.calc_energy(target_idx)
        old_pos = self.system.positions[target_idx].copy()
        self.system.target_clust_idx = self.system.find_target_cluster()

        # Calculate wnew for the new configuration
        nrb = self.system.config.n_rosenbluth_trials
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):            
            r = np.cbrt(np.random.rand() * (self.system.config.upper_cutoff**3 - self.system.config.lower_cutoff**3) + self.system.config.lower_cutoff**3)
            # Uniform sampling on the sphere for direction
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian coordinates
            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_pos = (self.system.positions[anchor_idx] + np.array([x, y, z])) % (self.system.box_length)
            
            self.system.positions[target_idx] = new_pos
            new_energy = self.system.calc_energy(target_idx)
            w = np.exp(-new_energy / self.system.kT)
            if np.isnan(w) or np.isinf(w):
                w = 0
                wnew += 0
            else:
                wnew += w

            rosenbluth_weights.append((w, new_energy, new_pos))
            self.system.positions[target_idx] = old_pos

        if wnew == 0:
            self.rejections += 1
            return

        # Select one configuration based on Rosenbluth weights
        wnew_valid = sum(weight for weight, _, _ in rosenbluth_weights)
        rosenbluth_weights_norm = [(weight / wnew_valid, d, pos) for weight, d, pos in rosenbluth_weights]
        _, new_energy, selected_pos = rosenbluth_weights[np.random.choice(range(len(rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.system.positions[target_idx] = selected_pos

        wold = np.exp(-(old_energy) / self.system.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            old_energy_out = new_energy
            
            old_pos_out = self.system.positions[target_idx_out].copy()
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_pos_out, new_pos_out) <= self.system.config.upper_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            
            self.system.positions[target_idx_out] = new_pos_out.copy()
            new_energy_out = self.system.calc_energy(target_idx_out)
            w = np.exp(-new_energy_out / self.system.kT)

            wold += w
            self.system.positions[target_idx_out] = old_pos_out
            
        if self.system.bias is not None:
            self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy
        self.system.energy += delta_energy
        self.system.bias_energy += bias_energy

        Nout = len([i for i in range(self.system.num_particles) if self.system.types[i] == part_type]) - Nin
        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((Nout - Nin) / (Nin + 1))
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            self.system.positions[target_idx] = old_pos
            self.system.energy -= delta_energy
            self.system.bias_energy -= bias_energy
            self.rejections += 1

class NVTInOutMove(Move):
    def __init__(self, system):
        '''Initialize NVT in-out move for cluster nucleation with AVBMC bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.config.upper_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3
    
    def attempt_move(self, anchor_idx, Nin_idx, part_type):
        '''Attempt NVT move to remove particle from target cluster to bulk with nucleation bias'''
        self.attempts += 1
        Nin = len(Nin_idx)

        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return

        target_idx = np.random.choice(Nin_idx)
        old_pos = self.system.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        regen = True
        clust_pos = np.asarray([self.system.positions[i] for i in self.system.target_clust_idx])
        while regen:
            distances = np.linalg.norm(clust_pos - new_pos, axis=1)
            if np.all(distances > self.system.config.upper_cutoff):
                regen = False
            else:
                new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        # Store old cluster size for AVBMC calculation
        old_cluster_size = len(self.system.target_clust_idx) # might need to change this to half of cluster?
        delta_energy, bias_energy = self.system.calc_energy_delta(target_idx, new_pos, old_pos)
        # Nout = len([i for i in range(self.system.num_particles) if self.system.types[i] == self.system.types[target_idx]]) - Nin
        N_type = len([i for i in range(self.system.num_particles) if self.system.types[i] == part_type])
        n_type = len([i for i in self.system.target_clust_idx if self.system.types[i] == part_type])

        try:
            avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.system.kT)*self.Vout/self.Vin*(Nin)/(N_type-n_type+1)*(old_cluster_size/(old_cluster_size-1))
        except:
            avbmc_energy = 0
        acc_prob = min(1, avbmc_energy)
        if np.random.rand() >= acc_prob:
            # Reject move - position is already back at old_pos from calc_energy_delta
            self.rejections += 1
        else:
            # Accept move - update position and system energy
            self.system.positions[target_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class NVTOutInMove(Move):
    def __init__(self, system):
        '''Initialize NVT out-in move for cluster growth with Rosenbluth sampling and nucleation bias'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.config.upper_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_cutoff**3)
        self.Vout = self.system.box_length**3
    
    def attempt_move(self, anchor_idx, Nin_idx, part_type):
        '''Attempt NVT move to insert bulk particle into target cluster using Rosenbluth weighting'''
        self.attempts += 1
        Nin = len(Nin_idx)

        target_idx = np.random.randint(self.system.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0) or (target_idx in self.system.target_clust_idx):
            target_idx = np.random.randint(self.system.num_particles)

        old_energy = self.system.calc_energy(target_idx)
        old_pos = self.system.positions[target_idx].copy()

        # Calculate wnew for the new configuration
        nrb = self.system.config.n_rosenbluth_trials
        wnew = 0
        rosenbluth_weights = []
        for _ in range(nrb):  
            r = np.cbrt(np.random.rand() * (self.system.config.upper_cutoff**3 - self.system.config.lower_cutoff**3) + self.system.config.lower_cutoff**3)
            # Uniform sampling on the sphere for direction
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            
            # Convert to Cartesian coordinates
            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_pos = (self.system.positions[anchor_idx] + np.array([x, y, z])) % (self.system.box_length)
            
            self.system.positions[target_idx] = new_pos
            new_energy = self.system.calc_energy(target_idx)
            w = np.exp(-new_energy / self.system.kT)
            if np.isnan(w) or np.isinf(w):
                w = 0
                wnew += 0
            else:
                wnew += w
            rosenbluth_weights.append((w, new_energy, new_pos))
            self.system.positions[target_idx] = old_pos

        if wnew == 0:
            self.rejections += 1
            return

        # Select one configuration based on Rosenbluth weights
        rosenbluth_weights_norm = [(weight / wnew, d, pos) for weight, d, pos in rosenbluth_weights]
        _, new_energy, selected_pos = rosenbluth_weights[np.random.choice(range(len(rosenbluth_weights)), p=[weight for weight, d, pos in rosenbluth_weights_norm])]
        self.system.positions[target_idx] = selected_pos

        # Calculate wold for the original configuration
        wold = np.exp(-(old_energy) / self.system.kT)  # Initial weight for SwapPart in the original position
        for _ in range(nrb - 1):  # Remaining trials
            target_idx_out = target_idx
            
            old_pos_out = self.system.positions[target_idx_out].copy()
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_pos_out, new_pos_out) <= self.system.config.upper_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            
            self.system.positions[target_idx_out] = new_pos_out
            new_energy_out = self.system.calc_energy(target_idx_out)
            w = np.exp(-new_energy_out / self.system.kT)

            wold += w
            self.system.positions[target_idx_out] = old_pos_out

        self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
        if self.system.bias is not None:
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy
        self.system.energy += delta_energy
        self.system.bias_energy += bias_energy

        # Nout = len([i for i in range(self.system.num_particles) if self.system.types[i] == self.system.types[target_idx]]) - Nin
        N_oppo = len([i for i in range(self.system.num_particles) if self.system.types[i] == part_type])
        n_oppo = len([i for i in self.system.tmp_target_clust_idx if self.system.types[i] == part_type])
        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((N_oppo - n_oppo) / (Nin + 1)) * ((len(self.system.tmp_target_clust_idx)) / (len(self.system.tmp_target_clust_idx)+1))
        # print(wnew/wold, (self.Vin / self.Vout), (N_oppo - n_oppo) / (Nin + 1), ((len(self.system.tmp_target_clust_idx)) / (len(self.system.tmp_target_clust_idx)+1)))
        acc_prob = min(1, avbmc_energy)
        # print(acc_prob)
        if np.random.rand() >= acc_prob:
            self.system.positions[target_idx] = old_pos
            self.system.energy -= delta_energy
            self.system.bias_energy -= bias_energy
            self.rejections += 1
