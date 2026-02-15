# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Files to Ignore

When working in this repository, ignore these files/directories:
- `test/` - Contains example outputs, not part of core codebase
- `inputs/nacl_us/` - Large input files, read-only data
- `potentials/` - Tabulated data files, should not be modified
- `bias/` - Bias potential files, typically read-only
- `*.log`, `*.out`, `*.xyz` - Output files from simulations

## Overview

This is a Monte Carlo simulation package for nucleation studies in continuous coordinate space. The code simulates speciation and nucleation of ions (specifically NaCl) represented as spherical particles using tabulated potentials.

## Core Architecture

The simulation system is built around three main Python modules:

- **simulation.py**: Main driver that handles simulation setup, execution, and output. Contains the `Simulation` class that orchestrates the Monte Carlo moves and manages parallel execution.
- **system.py**: Core physics implementation in the `System` class. Handles particle positions, energy calculations, cluster identification, and Monte Carlo move acceptance/rejection.
- **utils.py**: Utility classes including `PMF` (potential of mean force), `Particle`, and `Bias` for handling tabulated potentials and biasing schemes.

## Running Simulations

### Basic Commands

```bash
# Run a simple simulation
python simulation.py -config configs/25mM_nacl_config_adapUS_JC.txt -jobname test_run

# Run with multiple processors
python simulation.py -np 20 -jobname parallel_run -config configs/100mM_nacl_config.txt

# Run adaptive umbrella sampling
python simulation.py -adaptive -config configs/25mM_nacl_config_adapUS_JC.txt -jobname adaptive_run

# Get help on command line options
python simulation.py -h
```

### Umbrella Sampling Workflows

For umbrella sampling simulations:
1. Generate input structures: Modify and run `US_files/gen_US_inputs.py`
2. Set up template config file in `configs/nacl_us/`
3. Run the umbrella sampling driver: `bash US_files/run_us.sh`
4. Clean and combine trajectory data: `bash US_files/clean_colvars.sh`

## Configuration Files

Configuration files use a simple key-value format. Key parameters include:
- `box_length`: Simulation box size
- `num_particles`: Total number of particles
- `equil_steps`/`prod_steps`: Equilibration and production step counts
- `ff_file`: Tabulated potential file (stored in `potentials/`)
- `input_file`: Starting configuration (stored in `inputs/`)
- Move rates: `avbmc_rate`, `nvt_rate`, `translation_rate`, `swap_rate`
- Cluster cutoffs: `upper_cutoff`, `lower_cutoff`, `clust_cutoff`

## Directory Structure

- `configs/`: Configuration files for different systems/concentrations
- `potentials/`: Tabulated potential files (PMF data)
- `inputs/`: Starting configuration files (xyz format)
- `bias/`: Bias potential files for umbrella sampling
- `US_files/`: Scripts for umbrella sampling workflows
- `test/`: Example output directory structure

## Monte Carlo Moves

The system implements several move types:
- Translation moves
- Type-swap moves 
- AVBMC (Aggregation-Volume-Bias Monte Carlo) in/out moves
- NVT nucleation moves with Rosenbluth sampling

## Output Files

Simulations generate several output files:
- `E-XX.log`: Energy trajectories
- `traj-XX.xyz`: Coordinate trajectories
- `clusters-XX.out`: Cluster size distributions
- `target_cluster-XX.out`: Target cluster information
- `colvar_X.out`: Collective variable data for umbrella sampling
- `stats-XX.log`: Move acceptance statistics

## Cluster Analysis

The code uses Stillinger cluster analysis to identify nuclei. Particles within `clust_cutoff` distance are considered part of the same cluster. The target cluster (around particle 0) is tracked throughout the simulation.

## Testing

No automated test suite is present. Validate simulations by checking:
- Energy conservation in NVE runs
- Proper acceptance rates (~50-60% for translations)
- Cluster size distributions match expected behavior
- Output file consistency across parallel runs