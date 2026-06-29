#!/usr/bin/env python
"""Compute a 1D free energy profile (PMF) from umbrella-sampling windows with MBAR.

Adapted from the pymbar umbrella-sampling FES example.

Each window lives in <dir>/<N>mer/ where N is the harmonic bias center (cluster
size). The CV samples are read from colvar_<N>.log column 2 (cluster size); the
bias_energy column is ignored (buggy) and the harmonic bias is recomputed as
    u(x) = 0.5 * k * (x - center)^2
Everything is already in units of kT, so beta = 1 and the PMF comes out in kT.
"""
import argparse
import glob
import os
import re

import numpy as np
import pymbar
from pymbar import timeseries


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="../../NaCl_jobs/100mM_nacl_US/",
                   help="directory containing the <N>mer/ window folders")
    p.add_argument("-k", "--k", type=float, required=True,
                   help="harmonic spring constant in kT (u = 0.5*k*dx^2)")
    p.add_argument("--equil", type=int, default=0,
                   help="number of initial samples to discard per window")
    p.add_argument("--smin", type=float, default=-np.inf,
                   help="minimum cluster size to include in the analysis")
    p.add_argument("--smax", type=float, default=np.inf,
                   help="maximum cluster size to include in the analysis")
    p.add_argument("--no-subsample", action="store_true",
                   help="skip correlation-time subsampling")
    p.add_argument("--out", default="pmf.dat", help="output PMF file")
    return p.parse_args()


def read_colvar(path):
    """Return the CV (cluster size) samples from a colvar log, column 2."""
    vals = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.append(float(line.split()[1]))
    return np.array(vals)


def main():
    args = parse_args()

    # ---- discover windows -------------------------------------------------
    dirs = []
    for d in sorted(glob.glob(os.path.join(args.dir, "*mer")),
                    key=lambda d: int(re.search(r"(\d+)mer$", d).group(1))):
        m = re.search(r"(\d+)mer$", os.path.basename(d))
        if m and glob.glob(os.path.join(d, "colvar_*.log")):
            dirs.append((float(m.group(1)), d))

    K = len(dirs)
    if K == 0:
        raise SystemExit(f"no windows found in {args.dir}")

    chi0_k = np.array([c for c, _ in dirs])     # spring centers (bin centers)
    K_k = np.full(K, args.k)                     # spring constants (kT)
    N_k = np.zeros(K, dtype=int)

    # ---- read CV data -----------------------------------------------------
    raw = []
    for k, (center, d) in enumerate(dirs):
        colvar = glob.glob(os.path.join(d, "colvar_*.log"))[0]
        x = read_colvar(colvar)[args.equil:]
        # drop samples more than 50% away from this window's center
        #x = x[np.abs(x - center) <= 0.50 * center]
        # restrict to the requested analysis range
        x = x[(x >= args.smin) & (x <= args.smax)]
        raw.append(x)
        print(f"window {k:2d}  center={center:5.0f}  N_raw={len(x)}")

    N_max = max(len(x) for x in raw)
    chi_kn = np.zeros([K, N_max])

    # ---- subsample to uncorrelated samples --------------------------------
    for k, x in enumerate(raw):
        if args.no_subsample or len(x) < 2 or np.ptp(x) == 0:
            idx = np.arange(len(x))
        else:
            g = timeseries.statistical_inefficiency(x)
            idx = timeseries.subsample_correlated_data(x, g=g)
        N_k[k] = len(idx)
        chi_kn[k, :N_k[k]] = x[idx]

    N_max = int(N_k.max())
    chi_kn = chi_kn[:, :N_max]

    # ---- bins: one per integer cluster size -------------------------------
    all_chi = np.concatenate([chi_kn[k, :N_k[k]] for k in range(K)])
    s_min, s_max = int(all_chi.min()), int(all_chi.max())
    bin_edges = np.arange(s_min - 0.5, s_max + 1.5, 1.0)
    centers_all = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # keep only bins that actually contain samples (empty bins break get_fes)
    counts, _ = np.histogram(all_chi, bins=bin_edges)
    bin_center_i = centers_all[counts > 0]

    # ---- reduced potentials (beta = 1, already in kT) ---------------------
    # unbiased reduced potential is 0 for every sample (common Hamiltonian)
    u_kn = np.zeros([K, N_max])
    u_kln = np.zeros([K, K, N_max])
    for k in range(K):
        for n in range(N_k[k]):
            dchi = chi_kn[k, n] - chi0_k
            u_kln[k, :, n] = (K_k / 2.0) * dchi ** 2

    chi_n = pymbar.utils.kn_to_n(chi_kn, N_k=N_k)

    # ---- MBAR free energy profile -----------------------------------------
    fes = pymbar.FES(u_kln, N_k, verbose=True)
    fes.generate_fes(u_kn, chi_n, fes_type="histogram",
                     histogram_parameters={"bin_edges": bin_edges})
    results = fes.get_fes(bin_center_i, reference_point="from-lowest",
                          uncertainty_method="analytical")
    f_i = results["f_i"]
    df_i = results["df_i"]

    # ---- output -----------------------------------------------------------
    out = np.column_stack([bin_center_i, f_i, df_i])
    np.savetxt(args.out, out, header="size  PMF[kT]  dPMF[kT]", fmt="%g")
    print(f"\nwrote {args.out}")
    print(f"{'size':>6s} {'PMF[kT]':>9s} {'dPMF[kT]':>9s}")
    for s, f, df in out:
        print(f"{s:6.0f} {f:9.3f} {df:9.3f}")


if __name__ == "__main__":
    main()
