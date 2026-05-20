import numpy as np
import argparse
import os
import glob
from utils import Bias, logger

def load_cluster_data(sim_dir, cluster_type='target_cluster', num_chains=None):
    '''Load cluster size data from output files for all markov chains'''
    if cluster_type == 'target_cluster':
        pattern = f'{sim_dir}/target_cluster-*.out'
    elif cluster_type == 'clusters':
        pattern = f'{sim_dir}/clusters-*.out'
    else:
        raise ValueError(f"cluster_type must be 'target_cluster' or 'clusters', got {cluster_type}")

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No cluster files found matching pattern: {pattern}")

    if num_chains is not None:
        files = files[:num_chains]

    logger.info(f"Found {len(files)} cluster files")

    all_sizes = []
    for f in files:
        sizes = load_single_file(f, cluster_type)
        all_sizes.extend(sizes)
        logger.info(f"Loaded {len(sizes)} data points from {os.path.basename(f)}")

    return np.array(all_sizes)

def load_single_file(filepath, cluster_type):
    '''Parse cluster size data from a single output file'''
    sizes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if cluster_type == 'target_cluster':
                # Format: step size [indices]
                if len(parts) >= 2:
                    size = int(parts[1])
                    sizes.append(size)
            elif cluster_type == 'clusters':
                # Format: step [484   8] - array with spaces, not commas
                # dist[i] = number of clusters of size (i+1)
                if len(parts) >= 2:
                    # Remove brackets and convert to int array
                    array_str = ' '.join(parts[1:]).strip('[]')
                    dist = np.fromstring(array_str, dtype=int, sep=' ')
                    # Add all clusters: for each size, add that many entries
                    for size_idx, count in enumerate(dist):
                        cluster_size = size_idx + 1
                        sizes.extend([cluster_size] * count)
    return sizes

def calculate_bias(cluster_sizes, max_size=200, kT=0.596, output_file=None):
    '''Calculate bias potential from cluster size distribution using adaptive US method'''
    logger.info(f"Total samples: {len(cluster_sizes)}")
    logger.info(f"Min cluster size: {np.min(cluster_sizes)}")
    logger.info(f"Max cluster size: {np.max(cluster_sizes)}")

    # Create histogram
    dist = np.histogram(cluster_sizes, bins=np.arange(1, max_size+2))[0]
    logger.info(f"Histogram calculated with {len(dist)} bins")

    # Initialize bias and update with distribution
    bias = Bias(max_size=max_size, type='linear', kT=kT)
    potential = bias.update(dist)

    # Write outputs
    if output_file:
        base_dir = os.path.dirname(output_file)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Save histogram as space-separated values
        hist_file = output_file.replace('.npy', '_histogram.txt')
        with open(hist_file, 'w') as f:
            f.write(' '.join(map(str, dist)) + '\n')
        logger.info(f"Histogram saved to {hist_file}")

        # Save potential as space-separated values
        pot_file = output_file.replace('.npy', '_potential.txt')
        with open(pot_file, 'w') as f:
            f.write(' '.join(map(str, potential)) + '\n')
        logger.info(f"Potential saved to {pot_file}")

    return potential, dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate bias potential from cluster data')
    parser.add_argument('-sim_dir', type=str, required=True, help='Simulation directory containing cluster output files')
    parser.add_argument('-cluster_type', type=str, default='target_cluster', choices=['target_cluster', 'clusters'], help='Type of cluster file to read')
    parser.add_argument('-num_chains', type=int, default=8, help='Number of markov chains to read (default: all)')
    parser.add_argument('-max_size', type=int, default=10, help='Maximum cluster size for bias calculation')
    parser.add_argument('-kT', type=float, default=0.596, help='Temperature in kcal/mol (default: 0.596 = 300K)')
    parser.add_argument('-output', type=str, default='bias_potential.npy', help='Output file for bias potential')
    args = parser.parse_args()

    logger.info(f"Loading cluster data from: {args.sim_dir}")
    logger.info(f"Cluster type: {args.cluster_type}")

    cluster_sizes = load_cluster_data(args.sim_dir, args.cluster_type, args.num_chains)

    potential, dist = calculate_bias(cluster_sizes, args.max_size, args.kT, args.output)

    logger.info("Done!")

