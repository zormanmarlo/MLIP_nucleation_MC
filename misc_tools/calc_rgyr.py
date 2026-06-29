#!/usr/bin/env python3
"""Radius of gyration of the target cluster (the cluster containing atom 0) in an xyz file.

Standalone — uses only numpy. Reproduces the MC driver's definition
(BFS cluster from atom 0 within `cutoff`, PBC-unwrapped about the first member).

Usage:
    python calc_rgyr.py structure.xyz --box 202.49 [--cutoff 3.5]
"""
import argparse
from collections import deque

import numpy as np


def read_xyz(path):
    '''Read atom positions from a standard xyz file (ignores element labels)'''
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    return np.array([[float(v) for v in line.split()[1:4]] for line in lines[2:2 + n]])


def target_cluster(pos, cutoff, box):
    '''BFS from atom 0, collecting all atoms connected within cutoff (minimum image)'''
    visited, cluster, queue = set(), [], deque([0])
    while queue:
        i = queue.popleft()
        if i in visited:
            continue
        visited.add(i)
        cluster.append(i)
        d = pos - pos[i]
        d -= box * np.round(d / box)
        neighbors = np.where(np.linalg.norm(d, axis=1) < cutoff)[0]
        queue.extend(j for j in neighbors if j not in visited)
    return cluster


def calc_rgyr(pos, idx, box):
    '''Radius of gyration of cluster `idx`, unwrapped about its first member (0 for a monomer)'''
    if len(idx) < 2:
        return 0.0
    ref = pos[idx[0]]
    delta = pos[idx] - ref
    delta -= box * np.round(delta / box)
    unwrapped = ref + delta
    com = unwrapped.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((unwrapped - com) ** 2, axis=1))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz", help="input .xyz file")
    ap.add_argument("--box", type=float, required=True, help="cubic box length (Angstrom)")
    ap.add_argument("--cutoff", type=float, default=3.5, help="cluster bonding cutoff (Angstrom), default 3.5")
    args = ap.parse_args()

    pos = read_xyz(args.xyz)
    idx = target_cluster(pos, args.cutoff, args.box)
    print(f"target cluster size: {len(idx)}")
    print(f"radius of gyration:  {calc_rgyr(pos, idx, args.box):.4f} Angstrom")
