#!/usr/bin/env python
"""Compute a 1D free energy profile (PMF) from umbrella-sampling windows with WHAM.

Thin wrapper around the Grossfield WHAM binary
(/gscratch/cheme/mzorman/03_misc/wham/bin/wham), driven the same way as mbar_pmf.py:
same window layout, same CLI, same output format.

Each window lives in <dir>/<N>mer/ where N is the harmonic bias center (cluster size).
The CV samples are read from colvar_<N>.log column 2 (cluster size); the bias_energy
column is ignored and the harmonic bias is recomputed by WHAM as
    u(x) = 0.5 * k * (x - center)^2
We run WHAM with `units lj` (k_B = 1) and temperature 1.0, so beta = 1 and the PMF
comes out in kT -- exactly matching mbar_pmf.py. The spring constant convention is
identical (WHAM uses 0.5*k*dx^2), so -k passes through unchanged.
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile
import shutil

import numpy as np

WHAM_DEFAULT = "/gscratch/cheme/mzorman/03_misc/wham/bin/wham"


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
    p.add_argument("--maxdev", type=float, default=np.inf,
                   help="discard samples more than this many cluster sizes from "
                        "the window center (|x - center| > maxdev)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="number of WHAM Monte Carlo bootstrap trials for error bars "
                        "(0 = off, dPMF column written as 0)")
    p.add_argument("--seed", type=int, default=12345,
                   help="random seed for the bootstrap trials")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="WHAM convergence tolerance")
    p.add_argument("--wham", default=WHAM_DEFAULT, help="path to the wham binary")
    p.add_argument("--out", default="pmf_wham.dat", help="output PMF file")
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
        if not (m and glob.glob(os.path.join(d, "colvar_*.log"))):
            continue
        center = float(m.group(1))
        # smin/smax also restrict which window centers are loaded
        if center < args.smin or center > args.smax:
            continue
        dirs.append((center, d))

    if not dirs:
        raise SystemExit(f"no windows found in {args.dir}")

    # ---- read CV data -----------------------------------------------------
    raw = []
    for k, (center, d) in enumerate(dirs):
        colvar = glob.glob(os.path.join(d, "colvar_*.log"))[0]
        x = read_colvar(colvar)[args.equil:]
        x = x[(x >= args.smin) & (x <= args.smax)]
        # drop samples too far from this window's center
        x = x[np.abs(x - center) <= args.maxdev]
        raw.append((center, x))
        print(f"window {k:2d}  center={center:5.0f}  N_raw={len(x)}")

    # ---- write WHAM inputs into a temp dir --------------------------------
    workdir = tempfile.mkdtemp(prefix="wham_")
    meta_path = os.path.join(workdir, "metadata.txt")
    free_path = os.path.join(workdir, "free.dat")
    with open(meta_path, "w") as meta:
        for center, x in raw:
            if len(x) == 0:
                continue
            ts_path = os.path.join(workdir, f"win_{int(center)}.dat")
            np.savetxt(ts_path, np.column_stack([np.arange(len(x)), x]), fmt="%g")
            # filename  loc_center  spring_const  (k passes through unchanged)
            meta.write(f"{ts_path} {center:g} {args.k:g}\n")

    # ---- histogram params: one bin per integer cluster size ---------------
    all_chi = np.concatenate([x for _, x in raw if len(x)])
    s_min, s_max = int(all_chi.min()), int(all_chi.max())
    hist_min = s_min - 0.5
    hist_max = s_max + 0.5
    num_bins = s_max - s_min + 1
    numpad = 0

    # ---- run WHAM ---------------------------------------------------------
    cmd = [args.wham, "units", "lj",
           f"{hist_min:g}", f"{hist_max:g}", str(num_bins),
           f"{args.tol:g}", "1.0", str(numpad), meta_path, free_path]
    if args.bootstrap > 0:
        cmd += [str(args.bootstrap), str(args.seed)]
    print("\n$ " + " ".join(cmd) + "\n")
    try:
        subprocess.run(cmd, check=True)

        # ---- parse free.dat -----------------------------------------------
        # columns: coord  Free  +/-dF  Prob  +/-dP   (header lines start with #)
        data = np.loadtxt(free_path, comments="#")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    coord = data[:, 0]
    free = data[:, 1]
    dfree = data[:, 2] if data.shape[1] > 2 else np.zeros_like(free)

    # drop empty bins (WHAM writes inf/nan where there are no samples)
    good = np.isfinite(free)
    coord, free, dfree = coord[good], free[good], dfree[good]

    # reference from-lowest (matches mbar_pmf.py)
    free = free - free.min()

    # ---- output -----------------------------------------------------------
    out = np.column_stack([coord, free, dfree])
    np.savetxt(args.out, out, header="size  PMF[kT]  dPMF[kT]", fmt="%g")
    print(f"\nwrote {args.out}")
    print(f"{'size':>6s} {'PMF[kT]':>9s} {'dPMF[kT]':>9s}")
    for s, f, df in out:
        print(f"{s:6.0f} {f:9.3f} {df:9.3f}")


if __name__ == "__main__":
    main()
