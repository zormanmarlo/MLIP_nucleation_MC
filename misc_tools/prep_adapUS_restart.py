import numpy as np
import argparse
import glob
import yaml
import os

parser = argparse.ArgumentParser(description='Prep adapUS restart: create adapUS_state.npz and reset checkpoint steps to 0')
parser.add_argument('-job_dir', required=True)
args = parser.parse_args()

with open(f'{args.job_dir}/config.yaml') as f:
    config = yaml.safe_load(f)

prod_steps = config['prod_steps']

state_path = f'{args.job_dir}/adapUS_state.npz'
np.savez(state_path, iteration=np.array(0), prod_steps=np.array(prod_steps))
print(f"Created adapUS_state.npz (iteration=0, prod_steps={prod_steps})")

for ckpt in sorted(glob.glob(f'{args.job_dir}/checkpoint-*.npz')):
    d = np.load(ckpt, allow_pickle=True)
    old_step = int(d['step'])
    rng = d['rng_state']
    rng_state = rng.item() if rng.size == 1 else tuple(rng)
    rng_arr = np.empty(1, dtype=object)
    rng_arr[0] = rng_state
    np.savez(ckpt,
             step=np.array(0),
             positions=d['positions'],
             energy=d['energy'],
             bias_energy=d['bias_energy'],
             rng_state=rng_arr)
    print(f"Reset {os.path.basename(ckpt)}: step {old_step} → 0")
