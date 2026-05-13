import os
import sys
import shutil
import argparse

import numpy as np
import pickle as pkl
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from system import System
from utils import *

class Simulation:
    def __init__(self, config_file, jobname, seed=None, ID=0, bias=None):
        self.parameters = self.read_config(config_file)
        self.ID = str(ID).zfill(2)
        if seed is not None:
            self.parameters["seed"] = self.parameters["seed"] + seed
        np.random.seed(self.parameters["seed"])
        
        self.system = System(self.parameters["box_length"], self.parameters["num_particles"], s=self.parameters["seed"], target_max=self.parameters["max_target"], bias_path=bias)
        self.system.init()

        self.clust_sizes, self.target_clust = self.system.find_clusters()
        self.target_sizes = [len(self.target_clust)]
        self.jobname = jobname

    def read_config(self, config_file):
        with open(config_file, 'r') as f:
            parameters = {}
            for line in f:
                tmp = line.split()
                key, value = tmp[0], tmp[-1]
                if value.isdigit():
                    parameters[key] = int(value)
                else:
                    if value == "True":
                        parameters[key] = True
                    else:
                        parameters[key] = False
        return parameters
    
    def clean_dir(self):
        if os.path.exists(f'{self.jobname}/E-{self.ID}.log'):
            os.remove(f'{self.jobname}/E-{self.ID}.log')
        if os.path.exists(f'{self.jobname}/traj-{self.ID}.xyz'):
            os.remove(f'{self.jobname}/traj-{self.ID}.xyz')
        if os.path.exists(f'{self.jobname}/clusters-{self.ID}.out'):
            os.remove(f'{self.jobname}/clusters-{self.ID}.out')
        if os.path.exists(f'{self.jobname}/target_cluster-{self.ID}.out'):
            os.remove(f'{self.jobname}/target_cluster-{self.ID}.out')
        if os.path.exists(f'{self.jobname}/stats-{self.ID}.log'):
            os.remove(f'{self.jobname}/stats-{self.ID}.log')

    def write_output(self, step):
        with open(f'{self.jobname}/E-{self.ID}.log', 'a') as f:
            f.write(f'{step} {self.system.energy} {self.system.bias_energy}\n')
        with open(f'{self.jobname}/traj-{self.ID}.xyz', 'a') as f:
            total_num = self.system.num_particles + self.system.num_surf_particles
            f.write(f'{total_num}\n')
            f.write(f'  Step: {step}\n')
            for particle in self.system.particles:
                if particle.type == 0:
                    f.write(f'H {particle.position[0]:>6.2f} {particle.position[1]:>6.2f} {particle.position[2]:>6.2f}\n')
                elif particle.type == 1:
                    f.write(f'O {particle.position[0]:>6.2f} {particle.position[1]:>6.2f} {particle.position[2]:>6.2f}\n')
            for particle in self.system.surf_particles:
                if particle.type == 2:
                    f.write(f'Cl {particle.position[0]:>6.2f} {particle.position[1]:>6.2f} {particle.position[2]:>6.2f}\n')
                elif particle.type == 0:
                    f.write(f'H {particle.position[0]:>6.2f} {particle.position[1]:>6.2f} {particle.position[2]:>6.2f}\n')
                elif particle.type == 1:
                    f.write(f'O {particle.position[0]:>6.2f} {particle.position[1]:>6.2f} {particle.position[2]:>6.2f}\n')
        with open(f'{self.jobname}/clusters-{self.ID}.out', 'a') as f:
            clust_size_dist = np.histogram(self.clust_sizes, bins=np.arange(1, np.max(self.clust_sizes)+2))[0]
            f.write(f'{step} {clust_size_dist}\n')
        with open(f'{self.jobname}/target_cluster-{self.ID}.out', 'a') as f:
            f.write(f'{step} {len(self.target_clust)} {self.target_clust}\n')
        with open(f'{self.jobname}/stats-{self.ID}.log', 'a') as f:
            f.write(f'{step} {self.system.rejected_rates[0]/(self.system.attempts[0]+1)} {self.system.rejected_rates[1]/(self.system.attempts[1]+1)} {self.system.rejected_rates[2]/(self.system.attempts[2]+1)} {self.system.rejected_rates[3]/(self.system.attempts[3]+1)} {self.system.rejected_rates[4]/(self.system.attempts[4]+1)}\n')
        
        for i in range(len(self.system.rejected_rates)):
            self.system.rejected_rates[i] = 0
            self.system.attempts[i] = 0
        
    def step(self):
        # Perform NVT, AVBMC, or translation move
        # tmp = np.random.rand()
        # tmp = 0.0
        # Bath move (translation or AVBMC)
        # if tmp <= 0.80:
            tmp2 = np.random.rand()
            particle = np.random.randint(self.parameters["num_particles"])
            # translation move
            if tmp2 <= 0.75:
                self.system.translation(particle)
            # AVBMC move
            else:
                tmp3 = np.random.rand()
                Nin = self.system.calc_in(self.system.particles[particle])[0]
                # in -> out move
                if tmp3 <= 0.5 and Nin >= 1:
                    self.system.inout_AVBMC(particle)
                # out -> in move
                else:
                    self.system.outin_AVBMC(particle)

    def run(self, num_steps=None, equil=True):
        if num_steps is not None:
            self.parameters["num_steps"] = num_steps
        
        for s in range(1,self.parameters["num_steps"]):
            self.step()

            if s % self.parameters["internal_interval"] == 0:
                self.target_sizes.append(len(self.system.target_clust_idx))

            if s % self.parameters["output_interval"] == 0:
                # if equil:
                     # Niave translation acceptance rate adjustment
                trans_rate = self.system.rejected_rates[0]/(self.system.attempts[0]+1)
                current_max = self.system.max_displacement
                if trans_rate < 0.55:
                   self.system.max_displacement *= 1.1
                if trans_rate > 0.65:
                   self.system.max_displacement *= 0.9
                if self.system.max_displacement > 10 or self.system.max_displacement < 0.01:
                   self.system.max_displacement = current_max

                self.clust_sizes, self.target_clust = sef=self.system.find_clusters()
                self.write_output(s)

def test_hist(sim):
    it = int(sim.parameters["num_steps"]/sim.parameters["internal_interval"])

    # Exception for first run
    if len(sim.target_sizes) < it+1:
        return False
    
    target_counts = np.zeros((sim.parameters["max_target"]+2, len(sim.target_sizes)))
    for i, target_size in enumerate(sim.target_sizes):
        try:
            target_counts[:target_size, i] = np.arange(1, target_size+1)
        except:
            pass
    target_counts = np.cumsum(target_counts, axis=1)

    prev = np.mean(target_counts[:, 0:-it], axis=1)
    current = np.mean(target_counts[:, -it:], axis=1)
    cdf1 = prev / np.sum(prev)
    cdf2 = current / np.sum(current)

    return all(np.abs(cdf1 - cdf2) <= 0.25)

def equal_hist(dist):
    max_diff = np.max(np.abs(np.diff(dist)))
    if max_diff <= 0.05 * np.mean(dist):
        return True
    else:
        return False

def test_bias(potentials):
    if len(potentials) < 3:
        return False
    mse = np.sum((potentials[-2] - potentials[-1])**2)
    if mse < 0.05:
        return True
    else:
        return False

def convergence_run(sim, equil=False):
    conv = False
    i = 0 
    while not conv:
        i += 1
        sim.run(equil=equil)
        conv = test_hist(sim)
    return (i, sim)

def simple_run(sim):
    #try:
        sim.run()
        return sim
    #except Exception as e:
     #   return f"Error in simulation {sim}: {str(e)}"

if __name__ == "__main__":
    # Parse command line for settings
    parser = argparse.ArgumentParser(description='S-I-M-U-L-A-T-E')
    parser.add_argument('-np', type=int, default=1, help='Number of processors')
    parser.add_argument('-jobname', type=str, default='JOB', help='Name of the job')
    parser.add_argument('-config', type=str, default="config.txt", help="Configuration file")
    parser.add_argument('-US', type=bool, default=False, help="Run adaptive US or not")
    parser.add_argument('-seed', type=int, default=None, help="Random seed")
    parser.add_argument('-bias', type=str, default=None, help="Path to bias file")
    args = parser.parse_args()
    
    # Print jobname 
    print(args.jobname)
    # Set up directory and simulations
    if os.path.exists(args.jobname) and os.path.isdir(args.jobname):
        shutil.rmtree(args.jobname)
    try:
        os.makedirs(args.jobname, exist_ok=True)
        shutil.copy(args.config, f"{args.jobname}/config.txt")
        if args.bias is not None:
            shutil.copy(args.bias, f"{args.jobname}/bias.txt")
    except OSError as error:
        print(f'error creating directory -- exiting {args.jobname}: {error}')
    simulations = [Simulation(args.config, args.jobname, seed=i, ID=i, bias=args.bias) for i in range(args.np)]

    # Run simple simulation
    if not args.US:
        with mp.Pool(processes=args.np) as pool:
            simulations = pool.map(simple_run, simulations, True)
        # simple_run(simulations[0])
    # Run until potentials are converged
    else:
        # Run unbiased simulation
        with mp.Pool(processes=args.np) as pool:
            its_ran, simulations = zip(*pool.map(convergence_run, simulations))
        print("unbiased markov chains converged -- generating initial bias")
        
        # Run biased simulations, updating bias every time distribution converges until bias itself converges
        potential_history = []
        cont = True
        while cont:
            pkl.dump(simulations, open(f"{args.jobname}/system.pkl", "wb"))

            # sizes = np.concatenate([sim.target_sizes[-2 * sim.parameters["num_steps"] // sim.parameters["internal_interval"]:] for sim in simulations])
            sizes = np.concatenate([sim.target_sizes for sim in simulations])
            dist = np.histogram(sizes, bins=np.arange(1, simulations[0].parameters["max_target"]+2))[0]
            with open(f"{args.jobname}/histograms.out", "a") as file:
                file.write(f"{dist}\n")
            
            for sim in simulations:
                sim.system.bias.update(dist)
                sim.system.target_sizes = []
                sim.target_sizes = []
            potential_history.append(simulations[0].system.bias.bias)
            with open(f"{args.jobname}/potentials.out", "a") as file:
                file.write(f"{potential_history[-1]} - {its_ran}\n")
            
            with mp.Pool(processes=args.np) as pool:
                its_ran, simulations = zip(*pool.map(convergence_run, simulations))
            print("markov Chains converged -- updating bias")

            if equal_hist(dist):
                print("potential converged -- ending run")
                potential_history.append(simulations[0].system.bias.bias)
                
                # Save final system
                sizes = np.concatenate([sim.target_sizes[-2 * sim.parameters["num_steps"] // sim.parameters["output_interval"]:] for sim in simulations])
                dist = np.histogram(sizes, bins=np.arange(1, simulations[0].parameters["max_target"]+2))[0]
                with open(f"{args.jobname}/histograms.out", "a") as file:
                    file.write(f"{dist}\n")
                with open(f"{args.jobname}/potentials.out", "a") as file:
                    file.write(f"{potential_history[-1]}\n")
                
                pkl.dump(simulations, open(f"{args.jobname}/system.pkl", "wb"))
                cont = False





