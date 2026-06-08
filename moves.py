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
            self.rejections += 1
        else:
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
            self.rejections += 1
        else:
            self.system.positions[particle_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class InOutAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC in-out move with volume calculations for bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_bonded_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx):
        '''Attempt AVBMC move to remove particle from cluster to bulk solution'''
        self.attempts += 1

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)
        old_pos = self.system.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
        while self.system.calc_dist(old_pos, new_pos) <= self.system.config.upper_bonded_cutoff:
            new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        delta_energy, bias_energy = self.system.calc_energy_delta(target_idx, new_pos, old_pos)
        avbmc_energy = np.exp(np.clip(-(delta_energy+bias_energy)/self.system.kT,
                                -500, 500)) * self.Vout/self.Vin * (Nin)/(self.system.num_particles-Nin+1)
        acc_prob = min(1, avbmc_energy)

        if np.random.rand() >= acc_prob:
            self.rejections += 1
        else:
            self.system.positions[target_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class OutInAVBMCMove(Move):
    def __init__(self, system):
        '''Initialize AVBMC out-in move with volume calculations and Rosenbluth sampling'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_bonded_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx):
        '''Attempt AVBMC move to insert bulk particle into cluster using Rosenbluth weighting'''
        self.attempts += 1

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        target_idx = np.random.randint(self.system.num_particles)
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0):
            target_idx = np.random.randint(self.system.num_particles)

        old_energy = self.system.energy
        old_pos = self.system.positions[target_idx].copy()
        self.system.target_clust_idx = self.system.find_target_cluster()

        nrb = self.system.config.n_rosenbluth_trials
        new_energies = []
        new_positions = []

        for i in range(nrb):
            r = np.cbrt(np.random.rand() * (self.system.clust_cutoff**3 - self.system.config.lower_bonded_cutoff**3) + self.system.config.lower_bonded_cutoff**3)
            phi = 2 * np.pi * np.random.rand()
            cos_theta = 2 * np.random.rand() - 1
            sin_theta = np.sqrt(1 - cos_theta**2)

            x = r * sin_theta * np.cos(phi)
            y = r * sin_theta * np.sin(phi)
            z = r * cos_theta
            new_pos = (self.system.positions[anchor_idx] + np.array([x, y, z])) % (self.system.box_length)

            self.system.positions[target_idx] = new_pos
            trial_energy = self.system.calc_full_energy()
            new_energies.append(trial_energy)
            new_positions.append(new_pos.copy())
            self.system.positions[target_idx] = old_pos

        old_energies = [old_energy]
        for _ in range(nrb - 1):
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_pos, new_pos_out) <= self.system.config.upper_bonded_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

            self.system.positions[target_idx] = new_pos_out
            trial_energy = self.system.calc_full_energy()
            old_energies.append(trial_energy)
            self.system.positions[target_idx] = old_pos

        all_energies = new_energies + old_energies
        energy_ref = min(all_energies)

        wnew = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies)
        wold = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in old_energies)

        if wnew == 0:
            self.rejections += 1
            return

        weights_new = [np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies]
        weights_norm = [w / wnew for w in weights_new]
        selected_idx = np.random.choice(len(new_energies), p=weights_norm)
        new_energy = new_energies[selected_idx]
        selected_pos = new_positions[selected_idx]
        self.system.positions[target_idx] = selected_pos

        if self.system.bias is not None:
            self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy

        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_particles - Nin) / (Nin + 1))
        acc_prob = min(1, avbmc_energy)

        if np.random.rand() >= acc_prob:
            self.system.positions[target_idx] = old_pos
            self.rejections += 1
        else:
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class NVTInOutMove(Move):
    def __init__(self, system):
        '''Initialize NVT in-out move for cluster nucleation with AVBMC bias correction'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_bonded_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx, Nin_idx):
        '''Attempt NVT move to remove particle from target cluster to bulk with nucleation bias'''
        self.attempts += 1
        Nin = len(Nin_idx)

        Nin, Nin_idx = self.system.calc_in(anchor_idx)
        if Nin == 0 or (Nin == 1 and 0 in Nin_idx):
            self.rejections += 1
            return
        target_idx = np.random.choice(Nin_idx)
        old_pos = self.system.positions[target_idx].copy()

        new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
        regen = True
        self.system.target_clust_idx = self.system.find_target_cluster()
        clust_pos = np.asarray([self.system.positions[i] for i in self.system.target_clust_idx])
        while regen:
            distances = np.linalg.norm(clust_pos - new_pos, axis=1)
            if np.all(distances > self.system.config.upper_bonded_cutoff):
                regen = False
            else:
                new_pos = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

        old_cluster_size = len(self.system.target_clust_idx)
        delta_energy, bias_energy = self.system.calc_energy_delta(target_idx, new_pos, old_pos)

        try:
            avbmc_energy = np.exp(-(delta_energy+bias_energy)/self.system.kT)*self.Vout/self.Vin*(Nin)/(self.system.num_particles-old_cluster_size+1)*(old_cluster_size/(old_cluster_size-1))
        except:
            avbmc_energy = 0
        acc_prob = min(1, avbmc_energy)

        if np.random.rand() >= acc_prob:
            self.rejections += 1
        else:
            self.system.positions[target_idx] = new_pos
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy


class NVTOutInMove(Move):
    def __init__(self, system):
        '''Initialize NVT out-in move for cluster growth with Rosenbluth sampling and nucleation bias'''
        super().__init__(system)
        self.Vin = 4.0/3.0 * np.pi * (self.system.clust_cutoff**3) - 4.0/3.0 * np.pi * (self.system.config.lower_bonded_cutoff**3)
        self.Vout = self.system.box_length**3

    def attempt_move(self, anchor_idx, Nin_idx):
        '''Attempt NVT move to insert bulk particle into target cluster using Rosenbluth weighting'''
        self.attempts += 1
        Nin = len(Nin_idx)

        target_idx = np.random.randint(self.system.num_particles)
        self.system.target_clust_idx = self.system.find_target_cluster()
        while (target_idx in Nin_idx) or (target_idx == anchor_idx) or (target_idx == 0) or (target_idx in self.system.target_clust_idx):
            target_idx = np.random.randint(self.system.num_particles)

        old_energy = self.system.energy
        old_pos = self.system.positions[target_idx].copy()
        self.system.target_clust_idx = self.system.find_target_cluster()

        nrb = self.system.config.n_rosenbluth_trials
        new_energies = []
        new_positions = []

        for _ in range(nrb):
            displacement = np.round(((np.random.rand(3) - 0.5) * self.system.config.upper_bonded_cutoff * 2), 3)
            while ((sum(displacement**2) > self.system.config.upper_bonded_cutoff**2) or (sum(displacement**2) < self.system.config.lower_bonded_cutoff**2)):
                displacement = np.round(((np.random.rand(3) - 0.5) * self.system.config.upper_bonded_cutoff * 2), 3)

            new_pos = (self.system.positions[anchor_idx] + displacement) % (self.system.box_length)
            self.system.positions[target_idx] = new_pos
            trial_energy = self.system.calc_full_energy()
            new_energies.append(trial_energy)
            new_positions.append(new_pos.copy())
            self.system.positions[target_idx] = old_pos

        old_energies = [old_energy]
        for _ in range(nrb - 1):
            new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length
            while self.system.calc_dist(old_pos, new_pos_out) <= self.system.config.upper_bonded_cutoff:
                new_pos_out = np.round(((np.random.rand(3) - 0.5) * self.system.box_length * 2), 3) % self.system.box_length

            self.system.positions[target_idx] = new_pos_out
            trial_energy = self.system.calc_full_energy()
            old_energies.append(trial_energy)
            self.system.positions[target_idx] = old_pos

        all_energies = new_energies + old_energies
        energy_ref = min(all_energies)

        wnew = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies)
        wold = sum(np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in old_energies)

        if wnew == 0:
            self.rejections += 1
            return

        weights_new = [np.exp(np.clip(-(E - energy_ref) / self.system.kT, -700, 700)) for E in new_energies]
        weights_norm = [w / wnew for w in weights_new]
        selected_idx = np.random.choice(len(new_energies), p=weights_norm)
        new_energy = new_energies[selected_idx]
        selected_pos = new_positions[selected_idx]
        self.system.positions[target_idx] = selected_pos

        self.system.tmp_target_clust_idx = self.system.target_clust_idx.copy()
        if self.system.bias is not None:
            self.system.target_clust_idx = self.system.find_target_cluster()
            bias_energy = self.system.bias.denergy(len(self.system.target_clust_idx), len(self.system.tmp_target_clust_idx))
        else:
            bias_energy = 0.0

        delta_energy = new_energy - old_energy

        avbmc_energy = np.exp(-bias_energy/self.system.kT) * (wnew/wold) * (self.Vin / self.Vout) * ((self.system.num_particles - len(self.system.tmp_target_clust_idx)) / (Nin + 1)) * ((len(self.system.tmp_target_clust_idx)) / (len(self.system.tmp_target_clust_idx)+1))

        acc_prob = min(1, avbmc_energy)

        if np.random.rand() >= acc_prob:
            self.system.positions[target_idx] = old_pos
            self.rejections += 1
        else:
            self.system.energy += delta_energy
            self.system.bias_energy += bias_energy
