import os
import shutil
import argparse

import cProfile
import pstats

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
        if self.system.bias is not None and self.system.bias.type == 'harmonic':
            self.colvar_file = f'{self.output_dir}/colvar_{self.system.bias.center}.out'
        
        with open(self.stats_file, 'a') as f:
            move_headers = ' '.join(f'{name}_acceptance' for name in self.system.move_names)
            f.write(f'# step {move_headers}\n')
        
    def clean_dir(self):
        '''Remove all existing output files to ensure clean simulation start'''
        files_to_clean = [
            self.energy_file,
            self.traj_file,
            self.clusters_file,
            self.target_cluster_file,
            self.stats_file
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
        
        # Write trajectory output
        with open(self.traj_file, 'a') as f:
            f.write(f'  {self.config.num_particles}\n')
            f.write(f'  Step: {step}\n')
            for i, particle in enumerate(self.system.positions):
                atom_type = 'H' if self.system.types[i] == 0 else 'O'
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
        # for move in self.system.active_moves:
            # move.reset_stats()
        

def equal_hist(dist):
    '''Check if histogram distribution is sufficiently flat for adaptive umbrella sampling convergence'''
    max_diff = np.max(np.abs(np.diff(dist)))
    if max_diff <= 0.10 * np.mean(dist):
        return True
    else:
        return False

def production_run(sim):
    '''Execute production phase of simulation with periodic output writing'''
    for s in range(1, sim.config.parameters["prod_steps"]):
        sim.system.step()
        if s % sim.config.parameters["output_interval"] == 0:
            sim.write_output(s)
    return sim

def equilibration_run(sim):
    '''Execute equilibration phase with dynamic adjustment of translation move displacement'''
    for s in range(1, sim.config.parameters["equil_steps"]):
        sim.system.step()
    return sim

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
            equilibration_run(simulations[0])
            production_run(simulations[0])
            pr.disable()
            ps = pstats.Stats(pr).sort_stats('cumulative')
            ps.print_stats(10)
        else:
            with mp.Pool(processes=args.np) as pool:
                logger.info(f"Running {args.np} markov chains")
                logger.info("Running equilibration")
                simulations = pool.map(equilibration_run, simulations)
                logger.info("Running production")
                simulations = pool.map(production_run, simulations)

    # Run until potentials are converged
    else:
        # Run unbiased simulation
        with mp.Pool(processes=args.np) as pool:
            simulations = pool.map(equilibration_run, simulations, True)
            simulations = pool.map(production_run, simulations)

        # Run biased simulations, iteratively updating bias
        logger.info("Generating initial bias")
        cont = True
        orig_prod_steps = simulations[0].config.parameters["prod_steps"]
        current_it = 0
        while cont:
            current_it += 1
            pkl.dump(simulations, open(f"{args.jobname}/system.pkl", "wb"))

            cluster_counts = np.concatenate([sim.target_sizes for sim in simulations])
            dist = np.histogram(cluster_counts, bins=np.arange(1, simulations[0].config.parameters["max_target"]+2))[0]
            with open(f"{args.jobname}/histograms.out", "a") as file:
                file.write(f"{dist}\n")

            for sim in simulations:
                sim.system.bias.update(dist)
                # Reset data after each iteration, unclear if this is the best way to do this
                sim.system.target_sizes = []
                sim.target_sizes = []
                if all(dist > 0):
                    sim.system.config.parameters["prod_steps"] = sim.system.config.parameters["prod_steps"] + int(orig_prod_steps*0.2)
            potential = simulations[0].system.bias.bias
            with open(f"{args.jobname}/potentials.out", "a") as file:
                file.write(f"{potential}\n")

            with mp.Pool(processes=args.np) as pool:
                simulations = pool.map(production_run, simulations, True)
            print("updating bias")

            if equal_hist(dist):
                print(f"potential converged in {current_it} iterations -- ending run")
                potential = simulations[0].system.bias.bias

                # Save final system
                sizes = np.concatenate([sim.target_sizes[-2 * sim.config.parameters["num_steps"] // sim.config.parameters["output_interval"]:] for sim in simulations])
                dist = np.histogram(sizes, bins=np.arange(1, simulations[0].config.parameters["max_target"]+2))[0]
                with open(f"{args.jobname}/histograms.out", "a") as file:
                    file.write("FINAL HISTOGRAM:\n")
                    file.write(f"{dist}\n")
                with open(f"{args.jobname}/potentials.out", "a") as file:
                    file.write("FINAL POTENTIAL:\n")
                    file.write(f"{potential[-1]}\n")

                pkl.dump(simulations, open(f"{args.jobname}/system.pkl", "wb"))
                cont = False

