import os
import shutil
import argparse

import cProfile
import pstats
#import torch
#from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import pickle as pkl
import multiprocessing as mp

from system import System
from utils import *
from config import Config

class Simulation:
    def __init__(self, config_file, jobname, ID=0, path=".", multi_inputs=False):
        '''Initialize simulation with configuration, system setup, and output file paths'''
        self.config = Config(config_file)
        
        self.ID = str(ID).zfill(2)
        self.path = path
        self.jobname = jobname

        logger.info(f"Using seed from config file: {self.config.seed + ID}")
        np.random.seed(self.config.seed + ID)
        
        self.system = System(self.config, ID)
        self.system.init_positions(input_path=self.config.input_path, multi=multi_inputs)
        self.target_sizes = []


        # Pre-build file paths for cleaner code
        self.output_dir = f'{self.path}/{self.jobname}'
        self.stats_file = f'{self.output_dir}/stats-{self.ID}.log'
        self.energy_file = f'{self.output_dir}/E-{self.ID}.log'
        self.traj_file = f'{self.output_dir}/traj-{self.ID}.xyz'
        self.clusters_file = f'{self.output_dir}/clusters-{self.ID}.out'
        self.target_cluster_file = f'{self.output_dir}/target_cluster-{self.ID}.out'
        self.ca_proximity_file = f'{self.output_dir}/ca_proximity-{self.ID}.log'
        if self.system.bias is not None and self.system.bias.type == 'harmonic':
            self.colvar_file = f'{self.output_dir}/colvar_{self.system.bias.center}.out'

        # Set up Ca-Ca proximity logger
        self.setup_ca_proximity_logger()
        
        with open(self.stats_file, 'a') as f:
            move_headers = ' '.join(f'{name}_acceptance' for name in self.system.move_names)
            f.write(f'# step {move_headers}\n')

    def setup_ca_proximity_logger(self):
        '''Set up dedicated logger for Ca-Ca proximity alerts'''
        ca_logger = logging.getLogger(f'ca_proximity_{self.ID}')
        ca_logger.setLevel(logging.WARNING)
        ca_logger.propagate = False  # Don't propagate to root logger

        # Remove any existing handlers
        ca_logger.handlers = []

        # Create file handler for Ca-Ca proximity log
        file_handler = logging.FileHandler(self.ca_proximity_file, mode='w')
        file_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        ca_logger.addHandler(file_handler)

        # Store logger in system for moves to use
        self.system.ca_logger = ca_logger

    def clean_dir(self):
        '''Remove all existing output files to ensure clean simulation start'''
        files_to_clean = [
            self.energy_file,
            self.traj_file,
            self.clusters_file,
            self.target_cluster_file,
            self.stats_file,
            self.ca_proximity_file
        ]
        if hasattr(self, 'colvar_file'):
            files_to_clean.append(self.colvar_file)
            
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)

    def write_output(self, step):
        '''Write current simulation state to all output files including energy, trajectory, clusters, and statistics'''
        # Get current cluster information
        clust_sizes, target_clust = self.system.find_clusters()
        
        # Write collective variable output (for umbrella sampling)
        if hasattr(self, 'colvar_file'):
            with open(self.colvar_file, 'a') as f:
                f.write(f'{step} {len(target_clust)} {self.system.bias_energy}\n')
        
        # Write energy output
        with open(self.energy_file, 'a') as f:
            f.write(f'{step} {self.system.energy} {self.system.bias_energy}\n')
        
        # Write trajectory output (all atoms: Ca, C, O)
        with open(self.traj_file, 'a') as f:
            L = self.system.box_length
            f.write(f'  {self.config.num_particles}\n')
            f.write(f'  Lattice="{L} 0 0 0 {L} 0 0 0 {L}" Step: {step}\n')
            for i, particle in enumerate(self.system.positions):
                # Atom type mapping: 0=Ca, 1=C, 2=O
                atom_type_map = {0: 'Ca', 1: 'C', 2: 'O'}
                atom_type = atom_type_map.get(self.system.types[i], 'X')
                f.write(f'{atom_type} {particle[0]:>6.2f} {particle[1]:>6.2f} {particle[2]:>6.2f}\n')
        
        # Write cluster size distribution
        with open(self.clusters_file, 'a') as f:
            clust_size_dist = np.histogram(clust_sizes, bins=np.arange(1, np.max(clust_sizes)+2))[0]
            f.write(f'{step} {clust_size_dist}\n')
        
        # Write target cluster information
        with open(self.target_cluster_file, 'a') as f:
            f.write(f'{step} {len(target_clust)} {target_clust}\n')
        self.target_sizes.append(len(target_clust))
        
        # Write move acceptance statistics
        with open(self.stats_file, 'a') as f:
            rates = [move.get_acceptance_rate() for move in self.system.active_moves]
            rates_str = ' '.join(f'{rate:.4f}' for rate in rates)
            f.write(f'{step} {rates_str}\n')
        
        # Reset stats for active moves only
        for move in self.system.active_moves:
            move.reset_stats()
        

def equal_hist(dist):
    '''Check if histogram distribution is sufficiently flat for adaptive umbrella sampling convergence'''
    max_diff = np.max(np.abs(np.diff(dist)))
    if max_diff <= 0.10 * np.mean(dist):
        return True
    else:
        return False

def production_run(sim):
    '''Execute production phase of simulation with periodic output writing'''
    # Ensure energy is initialized (in case production_run is called directly)
    if sim.system.energy == 0.0:
        sim.system.energy = sim.system.calc_full_energy()
    for s in range(1, sim.config.parameters["prod_steps"]):
        sim.system.step(step_num=s)
        if s % sim.config.parameters["output_interval"] == 0:
            sim.write_output(s)
    return None  # Don't return sim - can't pickle after model loads

def production_run_adapUS(sim):
    '''Production run for adaptive US - reads bias from file, returns statistics and positions'''
    # Read updated bias from file if it exists
    bias_file = f'{sim.output_dir}/bias_potential.npy'
    if os.path.exists(bias_file):
        updated_bias = np.load(bias_file)
        sim.system.bias.bias = updated_bias

    # Initialize energy if needed
    if sim.system.energy == 0.0:
        sim.system.energy = sim.system.calc_full_energy()

    # Run simulation
    for s in range(1, sim.config.parameters["prod_steps"]):
        sim.system.step(step_num=s)
        if s % sim.config.parameters["output_interval"] == 0:
            sim.write_output(s)

    # Return statistics and state (picklable!)
    return {
        'target_sizes': sim.target_sizes,
        'positions': sim.system.positions.copy(),
        'energy': sim.system.energy,
        'ID': sim.ID
    }

def equilibration_run(sim):
    '''Execute equilibration phase with dynamic adjustment of translation move displacement'''
    sim.system.energy = sim.system.calc_full_energy()
    for s in range(1, sim.config.parameters["equil_steps"]):
        sim.system.step(step_num=s)
    return None  # Don't return sim - can't pickle after model loads

if __name__ == "__main__":
    # Parse command line for settings
    parser = argparse.ArgumentParser(description='?S-I-M-U-L-A-T-E?')
    parser.add_argument('-np', type=int, default=1, help='Number of processors')
    parser.add_argument('-jobname', type=str, default='JOB', help='Name of the job')
    parser.add_argument('-config', type=str, default="config.txt", help="Configuration file")
    parser.add_argument('-adapUS', action='store_true', help="Run adaptive US")
    parser.add_argument('-multi_inputs', action='store_true', help="Use multiple input files (will add jobnum to input filepath in config: input.txt -> input.00.txt, input.01.txt, etc.)")
    parser.add_argument('-path', type=str, default=".", help="Path to save output")
    args = parser.parse_args()

    logger.info("Starting job: "+str(args.jobname))

    # Set up directory and simulations
    if os.path.exists(args.path+"/"+args.jobname) and os.path.isdir(args.path+"/"+args.jobname):
        shutil.rmtree(args.path+"/"+args.jobname)
    try:
        os.makedirs(args.path+"/"+args.jobname, exist_ok=True)
        shutil.copy(args.config, f"{args.path}/{args.jobname}/config.txt")

    except OSError as error:
        logger.error(f'error creating directory -- exiting {args.path+"/"+args.jobname}: {error}')
    simulations = [Simulation(args.config, args.jobname, ID=i, path=args.path, multi_inputs=args.multi_inputs) for i in range(args.np)]
    
    # Run simple simulation
    if not args.adapUS:
        # Only pool if running more than one markov chain
        if args.np == 1:
            pr = cProfile.Profile()
            pr.enable()
            #scalene_profiler.start()

            #with profile(
            #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #    record_shapes=False,
            #    with_stack=False
            #) as prof:
            #    with record_function("equilibration"):
            equilibration_run(simulations[0])
            production_run(simulations[0])
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            #prof.export_chrome_trace("trace.json")
           # scalene_profiler.stop()

            #pr.disable()
            ps = pstats.Stats(pr).sort_stats('cumulative')
            ps.print_stats(100)
        else:
            with mp.Pool(processes=args.np) as pool:
                logger.info(f"Running {args.np} markov chains")
                logger.info("Running equilibration")
                pool.map(equilibration_run, simulations)
                logger.info("Running production")
                pool.map(production_run, simulations)

    # Run until potentials are converged
    else:
        # Run unbiased equilibration first production iteration
        with mp.Pool(processes=args.np) as pool:
            pool.map(equilibration_run, simulations)
            pool.map(production_run, simulations)

        # Initialize target_sizes for each simulation (collected from initial run)
        for sim in simulations:
            sim.target_sizes = []

        # Run biased simulations, iteratively updating bias
        logger.info("Generating initial bias")
        cont = True
        orig_prod_steps = simulations[0].config.parameters["prod_steps"]
        current_it = 0
        bias_file = f"{args.path}/{args.jobname}/bias_potential.npy"

        while cont:
            current_it += 1

            # Run production in workers
            with mp.Pool(processes=args.np) as pool:
                results = pool.map(production_run_adapUS, simulations)

            # Unpack results from workers and update simulation state
            for result in results:
                worker_id = int(result['ID'])
                simulations[worker_id].target_sizes = result['target_sizes']
                simulations[worker_id].system.positions = result['positions']
                simulations[worker_id].system.energy = result['energy']

            # Aggregate statistics
            cluster_counts = np.concatenate([sim.target_sizes for sim in simulations])
            dist = np.histogram(cluster_counts, bins=np.arange(1, simulations[0].config.parameters["max_target"]+2))[0]

            with open(f"{args.jobname}/histograms.out", "a") as file:
                file.write(f"{dist}\n")

            # Update bias in main process
            for sim in simulations:
                sim.system.bias.update(dist)
                sim.target_sizes = []  # Reset for next iteration
                if all(dist > 0):
                    sim.config.parameters["prod_steps"] = sim.config.parameters["prod_steps"] + int(orig_prod_steps*0.2)

            # Save updated bias to file for workers to read next iteration
            potential = simulations[0].system.bias.bias
            np.save(bias_file, potential)

            with open(f"{args.jobname}/potentials.out", "a") as file:
                file.write(f"{potential}\n")

            logger.info(f"Iteration {current_it}: Updated bias saved to {bias_file}")

            if equal_hist(dist):
                logger.info(f"Potential converged in {current_it} iterations -- ending run")
                potential = simulations[0].system.bias.bias

                # Save final outputs
                with open(f"{args.jobname}/histograms.out", "a") as file:
                    file.write("FINAL HISTOGRAM:\n")
                    file.write(f"{dist}\n")
                with open(f"{args.jobname}/potentials.out", "a") as file:
                    file.write("FINAL POTENTIAL:\n")
                    file.write(f"{potential}\n")

                # Save final bias potential
                np.save(f"{args.jobname}/final_bias_potential.npy", potential)
                cont = False

