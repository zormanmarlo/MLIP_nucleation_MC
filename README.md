# Monte Carlo Model for Simulation of Ion Aggregation and Nucleation

## Overview
A Monte Carlo code that simulates speciation and nucleation of ions in a continuous coordinate space. Ions are represented atomistically. Interactions are calculated with a physical prior + MLIP.
Framework is currently configured to calculate energy with LAMMPS and uses an Allegro model. Codebase is WIP with branches providing specific functionality.
 
## Files and Classes

- `simulation.py`: Main simulation driver class, handles I/O and parallelization
- `system.py`: Physics implementation and Monte Carlo engine, handles particle positions, types, clusters, etc
- `config.py`: Configuration file parser and validation

**utils.py**
- `PMF`: Potential of mean force class for calculating energies (legacy name)
  - Constructs and calls LAMMPS calls
- `Bias`: Umbrella sampling bias potential implementation
  - Supports harmonic and linear bias types
  - Handles adaptive bias updates for enhanced sampling
- Numba-compiled functions: `calc_energy_numba()`, `find_neighbors_numba()`, `interpolate_energy_numba()`

**moves.py**
- `Move`: Base class for Monte Carlo moves with statistics tracking
- `TranslationMove`: Random particle displacement within spherical constraint
- `SwapMove`: Random particle repositioning anywhere in box
- `InOutAVBMCMove`/`OutInAVBMCMove`: Aggregation-Volume-Bias Monte Carlo moves
- `NVTInOutMove`/`NVTOutInMove`: Nucleation moves with Rosenbluth sampling

## Simple example

Run a basic NaCl nucleation simulation:

```bash
# Run a short test simulation (single processor)
python simulation.py -config configs/test_config.txt -jobname test_run

# Run with multiple processors for better statistics
python simulation.py -np 4 -config configs/test_config.txt -jobname parallel_test

# Run with custom output directory
python simulation.py -config configs/test_config.txt -jobname my_sim -path ./results
```

**Output Files Generated:**
- `E-XX.log`: Energy vs time trajectories
- `traj-XX.xyz`: Particle coordinates
- `clusters-XX.log`: Cluster size distributions over time
- `target_cluster-XX.log`: Size of cluster around particle 0
- `stats-XX.log`: Monte Carlo move acceptance rates
