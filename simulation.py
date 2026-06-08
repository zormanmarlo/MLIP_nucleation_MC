import os
import shutil
import argparse
import faulthandler
faulthandler.enable()

import cProfile
import pstats

import numpy as np
import pickle as pkl
import multiprocessing as mp

from system import System
from utils import *
from config import Config

class Simulation:
    def __init__(self, config_file, jobname, ID=0, path=".", multi_inputs=False, restart=False):
        '''Initialize simulation with configuration, system setup, and output file paths'''
        self.config = Config(config_file)

        self.ID = str(ID).zfill(2)
        self.path = path
        self.jobname = jobname
        self.restart_step = 0

        logger.info(f"Using seed from config file: {self.config.seed + ID}")
        np.random.seed(self.config.seed + ID)

        self.system = System(self.config, ID)
        self.system.init_positions(input_path=self.config.input_path, multi=multi_inputs)
        self.target_sizes = []

        self.output_dir = f'{self.path}/{self.jobname}'
        self.stats_file = f'{self.output_dir}/stats-{self.ID}.log'
        self.energy_file = f'{self.output_dir}/E-{self.ID}.log'
        self.traj_file = f'{self.output_dir}/traj-{self.ID}.xyz'
        self.clusters_file = f'{self.output_dir}/clusters-{self.ID}.log'
        self.target_cluster_file = f'{self.output_dir}/target_cluster-{self.ID}.log'
        if self.system.bias is not None and self.system.bias.type == 'harmonic':
            self.colvar_file = f'{self.output_dir}/colvar_{self.system.bias.center}.log'
        if self.config.parameters['output_rcut']:
            self.rcut_file = f'{self.output_dir}/rcut-{self.ID}.log'
        if self.config.parameters['output_rcut_traj']:
            self.rcut_traj_file = f'{self.output_dir}/rcut_traj-{self.ID}.xyz'

        if restart:
            self._restore_from_checkpoint()
        else:
            # Write one-time headers for all per-frame data files
            self._append_line(self.energy_file, '# step energy bias_energy')
            self._append_line(self.clusters_file, '# step counts (column i = number of clusters of size i)')
            self._append_line(self.target_cluster_file, '# step size members...')
            move_headers = ' '.join(f'{name}_acceptance' for name in self.system.move_names)
            self._append_line(self.stats_file, f'# step {move_headers}')
            if hasattr(self, 'colvar_file'):
                self._append_line(self.colvar_file, '# step size bias_energy')
            if hasattr(self, 'rcut_file'):
                self._append_line(self.rcut_file, '# step rcut')

    @staticmethod
    def _append_line(path, line):
        '''Append a single line of text to a file, adding the trailing newline'''
        with open(path, 'a') as f:
            f.write(f'{line}\n')

    def save_checkpoint(self, step):
        '''Save simulation state to checkpoint file for later restart'''
        checkpoint_file = f'{self.output_dir}/checkpoint-{self.ID}.npz'
        rng_state = np.random.get_state()
        np.savez(
            checkpoint_file,
            step=np.array(step),
            positions=self.system.positions,
            energy=np.array(self.system.energy),
            bias_energy=np.array(self.system.bias_energy),
            rng_state=np.array(rng_state, dtype=object),
        )

    def load_checkpoint(self):
        '''Load checkpoint file and return its contents as a dict'''
        checkpoint_file = f'{self.output_dir}/checkpoint-{self.ID}.npz'
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_file}. "
                "Cannot restart without a checkpoint."
            )
        data = np.load(checkpoint_file, allow_pickle=True)
        return {
            'step': int(data['step']),
            'positions': data['positions'],
            'energy': float(data['energy']),
            'bias_energy': float(data['bias_energy']),
            'rng_state': data['rng_state'].item(),
        }

    def _restore_from_checkpoint(self):
        '''Apply checkpoint state to this simulation object'''
        ckpt = self.load_checkpoint()
        self.system.positions = ckpt['positions'].copy()
        self.system.energy = ckpt['energy']
        self.system.bias_energy = ckpt['bias_energy']
        np.random.set_state(ckpt['rng_state'])
        self.restart_step = ckpt['step']
        logger.info(f"[{self.ID}] Restarting from step {self.restart_step}")

    def clean_dir(self):
        '''Remove all existing output files to ensure clean simulation start'''
        files_to_clean = [
            self.energy_file,
            self.traj_file,
            self.clusters_file,
            self.target_cluster_file,
            self.stats_file,
        ]
        if hasattr(self, 'colvar_file'):
            files_to_clean.append(self.colvar_file)
        if hasattr(self, 'rcut_file'):
            files_to_clean.append(self.rcut_file)
        if hasattr(self, 'rcut_traj_file'):
            files_to_clean.append(self.rcut_traj_file)
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)

    def write_output(self, step):
        '''Write current simulation state to all output files including energy, trajectory, clusters, and statistics'''
        clust_sizes, target_clust = self.system.find_clusters()

        # Collective variable output (umbrella sampling)
        if hasattr(self, 'colvar_file'):
            self._append_line(self.colvar_file, f'{step} {len(target_clust)} {self.system.bias_energy}')

        # Energy output
        self._append_line(self.energy_file, f'{step} {self.system.energy} {self.system.bias_energy}')

        # Trajectory output
        with open(self.traj_file, 'a') as f:
            f.write(f'  {self.config.num_particles}\n')
            f.write(f'  Step: {step}\n')
            for i, particle in enumerate(self.system.positions):
                atom_type = 'Na' if self.system.types[i] == 0 else 'Cl'
                f.write(f'{atom_type} {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')

        # Cluster size distribution (space-separated integers)
        clust_size_dist = np.histogram(clust_sizes, bins=np.arange(1, np.max(clust_sizes)+2))[0]
        self._append_line(self.clusters_file, f"{step} {' '.join(map(str, clust_size_dist))}")

        # Target cluster (size, then space-separated member indices)
        self._append_line(self.target_cluster_file, f"{step} {len(target_clust)} {' '.join(map(str, target_clust))}")
        self.target_sizes.append(len(target_clust))

        # Move acceptance statistics
        rates = [move.get_acceptance_rate() for move in self.system.active_moves]
        rates_str = ' '.join(f'{rate:.4f}' for rate in rates)
        self._append_line(self.stats_file, f'{step} {rates_str}')

        # Rcut output (only when cluster is larger than monomer)
        rcut_minimum = 1
        if hasattr(self, 'rcut_file'):
            if len(target_clust) > rcut_minimum:
                self._append_line(self.rcut_file, f'{step} {self.system.calc_rcut()}')
        if hasattr(self, 'rcut_traj_file'):
            if len(target_clust) > rcut_minimum:
                rcut, translated_positions, particle_types = self.system.calc_rcut(coordinates=True)
                with open(self.rcut_traj_file, 'a') as f:
                    f.write(f'  {self.config.num_particles}\n')
                    f.write(f'  Step: {step} target_cluster_size: {len(target_clust)} Rcut: {rcut:.4f}\n')
                    for i, particle in enumerate(translated_positions):
                        atom_type = 'Na' if particle_types[i] == 0 else 'Cl'
                        f.write(f'{atom_type} {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')

        # Checkpoint after every output write
        self.save_checkpoint(step)


def equal_hist(dist):
    '''Check if histogram distribution is sufficiently flat for adaptive umbrella sampling convergence'''
    max_diff = np.max(np.abs(np.diff(dist)))
    if max_diff <= 0.10 * np.mean(dist):
        return True
    else:
        return False

def production_run(sim):
    '''Execute production phase of simulation with periodic output writing'''
    if sim.system.energy == 0.0:
        sim.system.energy = sim.system.calc_full_energy()
    for s in range(sim.restart_step + 1, sim.config.parameters["prod_steps"]):
        sim.system.step(step_num=s)
        if s % sim.config.parameters["output_interval"] == 0:
            sim.write_output(s)
    return None  # Don't return sim - can't pickle after model loads

def production_run_adapUS(sim):
    '''Production run for adaptive US - reads bias from file, returns statistics and positions'''
    bias_file = f'{sim.output_dir}/bias_potential.npy'
    if os.path.exists(bias_file):
        updated_bias = np.load(bias_file)
        sim.system.bias.bias = updated_bias

    if sim.system.energy == 0.0:
        sim.system.energy = sim.system.calc_full_energy()

    for s in range(sim.restart_step + 1, sim.config.parameters["prod_steps"]):
        sim.system.step(step_num=s)
        if s % sim.config.parameters["output_interval"] == 0:
            sim.write_output(s)

    # Reset restart_step so subsequent iterations run fully
    sim.restart_step = 0

    return {
        'target_sizes': sim.target_sizes,
        'positions': sim.system.positions.copy(),
        'energy': sim.system.energy,
        'ID': sim.ID
    }

def equilibration_run(sim):
    '''Execute equilibration phase (skipped on restart)'''
    if sim.restart_step > 0:
        return None
    sim.system.energy = sim.system.calc_full_energy()
    for s in range(1, sim.config.parameters["equil_steps"]):
        sim.system.step(step_num=s)
    return None  # Don't return sim - can't pickle after model loads

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='?S-I-M-U-L-A-T-E?')
    parser.add_argument('-np', type=int, default=1, help='Number of processors')
    parser.add_argument('-jobname', type=str, default='JOB', help='Name of the job')
    parser.add_argument('-config', type=str, default="config.yaml", help="Configuration file (.yaml format)")
    parser.add_argument('-adapUS', action='store_true', help="Run adaptive US")
    parser.add_argument('-restart', action='store_true', help="Restart from checkpoint files in the job directory")
    parser.add_argument('-multi_inputs', action='store_true', help="Use multiple input files")
    parser.add_argument('-path', type=str, default=".", help="Path to save output")
    args = parser.parse_args()

    logger.info("Starting job: "+str(args.jobname))

    job_dir = f"{args.path}/{args.jobname}"

    if not args.restart:
        if os.path.exists(job_dir) and os.path.isdir(job_dir):
            shutil.rmtree(job_dir)
        try:
            os.makedirs(job_dir, exist_ok=True)
            shutil.copy(args.config, f"{job_dir}/config.yaml")
        except OSError as error:
            logger.error(f'error creating directory -- exiting {job_dir}: {error}')
    else:
        logger.info(f"Restarting from existing job directory: {job_dir}")

    simulations = [
        Simulation(args.config, args.jobname, ID=i, path=args.path,
                   multi_inputs=args.multi_inputs, restart=args.restart)
        for i in range(args.np)
    ]

    if not args.adapUS:
        if args.np == 1:
            pr = cProfile.Profile()
            pr.enable()
            equilibration_run(simulations[0])
            production_run(simulations[0])
            pr.disable()
            ps = pstats.Stats(pr).sort_stats('cumulative')
            ps.print_stats(100)
        else:
            with mp.Pool(processes=args.np) as pool:
                logger.info(f"Running {args.np} markov chains")
                logger.info("Running equilibration")
                pool.map(equilibration_run, simulations)
                logger.info("Running production")
                pool.map(production_run, simulations)

    else:
        adapUS_state_file = f"{job_dir}/adapUS_state.npz"

        # Restore adapUS loop state on restart
        current_it = 0
        orig_prod_steps = simulations[0].config.parameters["prod_steps"]
        if args.restart and os.path.exists(adapUS_state_file):
            state = np.load(adapUS_state_file)
            current_it = int(state['iteration'])
            saved_prod_steps = int(state['prod_steps'])
            for sim in simulations:
                sim.config.parameters["prod_steps"] = saved_prod_steps
            logger.info(f"Resuming adapUS from iteration {current_it}, prod_steps={saved_prod_steps}")
        else:
            # Fresh start: run unbiased equil + initial production
            with mp.Pool(processes=args.np) as pool:
                pool.map(equilibration_run, simulations)
                pool.map(production_run, simulations)
            for sim in simulations:
                sim.target_sizes = []

        logger.info("Generating initial bias" if current_it == 0 else "Continuing adapUS bias iteration")
        cont = True
        bias_file = f"{job_dir}/bias_potential.npy"

        while cont:
            current_it += 1

            with mp.Pool(processes=args.np) as pool:
                results = pool.map(production_run_adapUS, simulations)

            for result in results:
                worker_id = int(result['ID'])
                simulations[worker_id].target_sizes = result['target_sizes']
                simulations[worker_id].system.positions = result['positions']
                simulations[worker_id].system.energy = result['energy']

            cluster_counts = np.concatenate([sim.target_sizes for sim in simulations])
            dist = np.histogram(cluster_counts, bins=np.arange(1, simulations[0].config.parameters["max_target"]+2))[0]

            with open(f"{job_dir}/histograms.out", "a") as file:
                file.write(f"{dist}\n")

            for sim in simulations:
                sim.system.bias.update(dist)
                sim.target_sizes = []
                if all(dist > 0):
                    sim.config.parameters["prod_steps"] = sim.config.parameters["prod_steps"] + int(orig_prod_steps*0.2)

            potential = simulations[0].system.bias.bias
            np.save(bias_file, potential)

            with open(f"{job_dir}/potentials.out", "a") as file:
                file.write(f"{potential}\n")

            # Save adapUS loop state for restart
            np.savez(adapUS_state_file,
                     iteration=np.array(current_it),
                     prod_steps=np.array(simulations[0].config.parameters["prod_steps"]))

            logger.info(f"Iteration {current_it}: Updated bias saved to {bias_file}")

            if equal_hist(dist):
                logger.info(f"Potential converged in {current_it} iterations -- ending run")
                potential = simulations[0].system.bias.bias

                with open(f"{job_dir}/histograms.out", "a") as file:
                    file.write("FINAL HISTOGRAM:\n")
                    file.write(f"{dist}\n")
                with open(f"{job_dir}/potentials.out", "a") as file:
                    file.write("FINAL POTENTIAL:\n")
                    file.write(f"{potential}\n")

                np.save(f"{job_dir}/final_bias_potential.npy", potential)
                cont = False
