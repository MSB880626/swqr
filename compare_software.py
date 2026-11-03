#!/usr/bin/env python3
"""
compare_software.py — comparación de tiempo y exactitud contra software existente.

Todos los métodos evalúan TODAS las ventanas (paso b = 1), que es lo que hacen
statsmodels y rollRegres; así los tiempos son comparables ventana por ventana.

  propio (swqr_spark, Numba):  qr (two-stacks / contador binario), qr-tree, qr-seq, ne
  statsmodels RollingOLS:      inv (ecuaciones normales con update/downdate en bucle Python),
                               inv con reset, lstsq (numpy.lstsq por ventana; solo si N·W lo permite)
  R rollRegres:                roll_regres.fit (QR con update/downdate de rango 1, LINPACK);
                               do_downdates = FALSE para ventana expansiva

Referencias de exactitud: lstsq con rcond=-1 en --err-points ventanas y mpmath (60 dígitos)
en --mp-points ventanas.

Uso (Colab):
  !pip -q install statsmodels numba mpmath
  !Rscript -e 'install.packages("rollRegres", repos="https://cloud.r-project.org")'
  !python compare_software.py --csv comparacion.csv
Requiere swqr_spark.py en la misma carpeta.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swqr_spark as S  # noqa: E402

FIELDS = ["mode", "software", "method", "N", "p", "W", "b", "level", "seed", "kappa_median", "nwin",
          "time_s", "us_per_window", "err_points", "err_median", "err_max", "mp_points",
          "err_mp_median", "err_mp_max", "note"]

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
d <- args[1]; N <- as.integer(args[2]); p <- as.integer(args[3]); W <- as.integer(args[4])
expanding <- args[5] == "1"
suppressPackageStartupMessages(library(rollRegres))
X <- matrix(readBin(file.path(d, "X.bin"), "double", n = N * p), nrow = N)
y <- readBin(file.path(d, "y.bin"), "double", n = N)
idx <- readBin(file.path(d, "idx.bin"), "integer", n = file.size(file.path(d, "idx.bin")) / 4)
t0 <- proc.time()[["elapsed"]]
if (expanding) {
  fit <- roll_regres.fit(X, y, width = W, do_downdates = FALSE)
} else {
  fit <- roll_regres.fit(X, y, width = W)
}
el <- proc.time()[["elapsed"]] - t0
B <- fit$coefs[idx + 1, , drop = FALSE]
writeBin(as.vector(t(B)), file.path(d, "B.bin"))
writeLines(format(el, digits = 10), file.path(d, "time.txt"))
"""


def have_rollregres():
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", "suppressPackageStartupMessages(library(rollRegres))"],
                       capture_output=True)
    return r.returncode == 0


def run_rollregres(X, y, W, pts, expanding):
    with tempfile.TemporaryDirectory() as d:
        X.T.astype(np.float64).tofile(os.path.join(d, "X.bin"))  # columna mayor para R
        y.astype(np.float64).tofile(os.path.join(d, "y.bin"))
        np.asarray(pts, dtype=np.int32).tofile(os.path.join(d, "idx.bin"))
        rs = os.path.join(d, "run.R")
        with open(rs, "w") as f:
            f.write(R_SCRIPT)
        r = subprocess.run(["Rscript", rs, d, str(len(y)), str(X.shape[1]), str(W),
                            "1" if expanding else "0"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        B = np.fromfile(os.path.join(d, "B.bin"), dtype=np.float64).reshape(len(pts), X.shape[1])
        t = float(open(os.path.join(d, "time.txt")).read())
    return B, t


def run_statsmodels(X, y, W, method, expanding, reset=None):
    from statsmodels.regression.rolling import RollingOLS

    t0 = time.perf_counter()
    # En statsmodels, expanding=True solo crece hasta `window` y luego desliza; para un
    # prefijo completo la ventana debe ser N, con min_nobs = número mínimo de filas.
    if expanding:
        mod = RollingOLS(y, X, window=len(y), expanding=True, min_nobs=W)
    else:
        mod = RollingOLS(y, X, window=W)
    res = mod.fit(method=method, params_only=True, reset=reset)
    t = time.perf_counter() - t0
    return np.asarray(res.params), t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", choices=["sliding", "expanding", "both"], default="both")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--windows", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--exp-min-rows", type=int, default=1000)
    ap.add_argument("--levels", type=float, nargs="+", default=[1e3, 1e4, 1e5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--block", type=int, default=1,
                    help="paso b entre ventanas evaluadas por los métodos propios; statsmodels y "
                         "rollRegres no pueden saltar filas y siempre procesan todas")
    ap.add_argument("--err-points", type=int, default=50)
    ap.add_argument("--mp-points", type=int, default=4)
    ap.add_argument("--sm-reset", type=int, default=1000, help="reset de statsmodels (filas) para 'inv-reset'")
    ap.add_argument("--sm-lstsq-max", type=float, default=2e9,
                    help="corre statsmodels lstsq solo si N·W <= este valor (cuesta O(N·W·p²))")
    ap.add_argument("--no-r", action="store_true")
    ap.add_argument("--no-statsmodels", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    use_r = not a.no_r and have_rollregres()
    if not a.no_r and not use_r:
        print("aviso: Rscript o rollRegres no disponibles; se omite rollRegres", file=sys.stderr)
    use_sm = not a.no_statsmodels
    if use_sm:
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            print("aviso: statsmodels no instalado; se omite", file=sys.stderr)
            use_sm = False
    S.numba_warmup()

    done = set()
    if a.resume and os.path.exists(a.csv):
        with open(a.csv) as f:
            done = {(r["mode"], r["software"], r["method"], int(r["N"]), int(r["W"]), int(r.get("b") or 1),
                     float(r["level"]), int(r["seed"])) for r in csv.DictReader(f)}
    new = not (a.resume and os.path.exists(a.csv))
    fh = open(a.csv, "w" if new else "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        wr.writeheader()
        fh.flush()

    modes = ["sliding", "expanding"] if a.mode == "both" else [a.mode]
    for level in a.levels:
        for seed in a.seeds:
            _, ch = S.arx_chunks(a.n, level=level, seed=seed)
            _, y_, X_ = (np.concatenate(v) for v in zip(*ch))
            A = np.ascontiguousarray(S.design(X_, y_, True))
            X, y = np.ascontiguousarray(A[:, :-1]), np.ascontiguousarray(A[:, -1])
            N, p = len(y), X.shape[1]
            print(f"datos N={N} p={p} level={level:.0e} seed={seed}", flush=True)
            cfgs = []
            if "sliding" in modes:
                cfgs += [("sliding", W) for W in a.windows if W <= N // 2 and W % a.block == 0]
            if "expanding" in modes:
                cfgs.append(("expanding", a.exp_min_rows))
            for mode, W in cfgs:
                exp = mode == "expanding"
                bb = a.block
                ends = np.array([e for e in S.aligned_ends(0, N - 1, bb) if e + 1 >= W])
                nwin = len(ends)
                pts = ends[np.unique(np.linspace(0, nwin - 1, min(a.err_points, nwin)).astype(int))]
                lo = (lambda e: 0) if exp else (lambda e: e - W + 1)
                kap = float(np.median([np.linalg.cond(X[lo(e):e + 1]) for e in pts[:: max(1, len(pts) // 8)]]))
                Bref = np.vstack([S.solve_lstsq(A[lo(e):e + 1])[0] for e in pts])
                mp_pts = pts[np.unique(np.linspace(0, len(pts) - 1, min(a.mp_points, len(pts))).astype(int))] \
                    if a.mp_points > 0 else np.array([], dtype=int)
                mp_ref = S.reference_mp(A, list(mp_pts), W, mode) if len(mp_pts) else {}
                base = dict(mode=mode, N=N, p=p, W=W, b=bb, level=level, seed=seed, kappa_median=kap, nwin=nwin)

                def emit(software, method, Bpts, t, note=""):
                    if (mode, software, method, N, W, bb, float(level), seed) in done:
                        return
                    err = S.rel_err(Bpts, Bref)
                    row = dict(base, software=software, method=method, time_s=t,
                               us_per_window=1e6 * t / nwin, err_points=len(pts),
                               err_median=float(np.nanmedian(err)), err_max=float(np.nanmax(err)),
                               mp_points=len(mp_pts), note=note)
                    if len(mp_pts):
                        pos = np.searchsorted(pts, mp_pts)
                        em = np.array([S.rel_err(Bpts[i], mp_ref[e])[0] for i, e in zip(pos, mp_pts)])
                        row.update(err_mp_median=float(np.nanmedian(em)), err_mp_max=float(np.nanmax(em)))
                    else:
                        row.update(err_mp_median=np.nan, err_mp_max=np.nan)
                    wr.writerow(row)
                    fh.flush()
                    print(f"  {mode:9s} W={W:>7d} b={bb:<3d} {software:11s} {method:9s} t={t:9.2f} s "
                          f"err_med={row['err_median']:.2e} err_max={row['err_max']:.2e}"
                          + (f" mp_max={row['err_mp_max']:.2e}" if len(mp_pts) else "") + (f"  [{note}]" if note else ""),
                          flush=True)

                def skip(software, method):
                    return (mode, software, method, N, W, bb, float(level), seed) in done

                # propios (Numba, b = 1)
                own = ["qr", "qr-tree", "ne"] if not exp else ["qr", "qr-seq", "ne"]
                for mth in own:
                    if skip("swqr", mth):
                        continue
                    t0 = time.perf_counter()
                    e_out, B, _ = (S.sliding_arrays(A, bb, W, mth, engine="numba") if not exp
                                   else S.expanding_arrays(A, bb, W, mth, engine="numba"))
                    t = time.perf_counter() - t0
                    emit("swqr", mth, B[np.searchsorted(e_out, pts)], t)
                    del B, e_out

                # statsmodels
                if use_sm:
                    sm = [("inv", None, "inv"), ("inv", a.sm_reset, "inv-reset")]
                    if not exp and N * W <= a.sm_lstsq_max:
                        sm.append(("lstsq", None, "lstsq"))
                    for method, reset, name in sm:
                        if skip("statsmodels", name):
                            continue
                        P, t = run_statsmodels(X, y, W, method, exp, reset)
                        note = "numpy.lstsq con rcond por omisión" if method == "lstsq" else ""
                        emit("statsmodels", name, P[pts], t, note)
                        del P

                # rollRegres
                rname = "qr-update" if exp else "qr-updown"
                if use_r and not skip("rollRegres", rname):
                    B, t = run_rollregres(X, y, W, pts, exp)
                    emit("rollRegres", rname, B, t)
            del A, X, y
    fh.close()
    print(f"resultados -> {a.csv}")


if __name__ == "__main__":
    main()
