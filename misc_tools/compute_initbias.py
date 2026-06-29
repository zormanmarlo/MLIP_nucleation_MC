import numpy as np
import argparse
import glob
import yaml
import os

parser = argparse.ArgumentParser(description='Compute initial adapUS bias from production run cluster data')
parser.add_argument('-job_dir', required=True)
args = parser.parse_args()

with open(f'{args.job_dir}/config.yaml') as f:
    config = yaml.safe_load(f)

max_size = config.get('max_target', 200)
kT = config.get('kT', 0.592)
bias_file = config.get('bias_file')

driver_dir = os.path.dirname(os.path.abspath(__file__))
if bias_file:
    init_bias = np.loadtxt(os.path.join(driver_dir, bias_file))
    print(f"Loaded initial bias ({len(init_bias)} entries) from {bias_file}")
    if len(init_bias) != max_size:
        print(f"WARNING: bias file has {len(init_bias)} entries but max_target={max_size}")
else:
    init_bias = np.zeros(max_size)
    print("No bias_file in config; starting from zero bias")

sizes = []
for f in sorted(glob.glob(f'{args.job_dir}/target_cluster-*.log')):
    with open(f) as fh:
        for line in fh:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    sizes.append(int(parts[1]))
sizes = np.array(sizes)
print(f"Loaded {len(sizes)} samples, cluster size range: {sizes.min()}–{sizes.max()}")

dist = np.histogram(sizes, bins=np.arange(1, max_size + 2))[0]
potential = init_bias.copy()
pivot = np.argmax(dist)
n_star = dist[pivot]
for i in range(len(dist)):
    if dist[i] > 0:
        potential[i] = init_bias[i] + kT * np.log(dist[i] / n_star)
    else:
        potential[i] = init_bias[pivot] + kT * np.log(1.0 / n_star)
potential -= potential[1]

np.save(f'{args.job_dir}/bias_potential.npy', potential)
np.savetxt(f'{args.job_dir}/initbias_histogram.txt', dist, fmt='%d')
np.savetxt(f'{args.job_dir}/initbias_potential.txt', potential)
print(f"Saved bias_potential.npy, initbias_histogram.txt, initbias_potential.txt to {args.job_dir}")
