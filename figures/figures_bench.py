#!/usr/bin/env python3
"""
figuras_bench.py — figuras del artículo a partir del CSV de `swqr_spark.py bench`.

Uso:
  python figuras_bench.py --csv bench_1e6_final.csv --out figuras [--exclude-levels 1e5]

Genera (PNG 300 dpi y PDF vectorial):
  fig1_tiempo_vs_W      sliding: tiempo total vs W (t_min; mediana y rango sobre niveles y semillas)
  fig2_error_vs_kappa   sliding: error relativo vs κ, paneles W mínimo y máximo
  fig3_error_vs_m       sliding: error de qr (two-stacks) y qr-tree vs m = W/b, por nivel de κ
  fig4_expanding        expanding: error vs κ (izq.) y tiempo por método (der.)
  tabla_resumen.csv     medianas de tiempo y error por método, W y nivel
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {"naive": "lstsq per window", "qr": "QR two-stacks", "qr-tree": "QR segment tree",
         "ne": "Normal equations", "ne-reset": "Normal equations with reset",
         "qr-seq": "Sequential QR"}
LABEL_EXP = {"qr": "QR binary counter", "qr-seq": "Sequential QR", "ne": "Normal equations"}
SHORT_EXP = {"qr": "QR tree", "qr-seq": "Sequential\nQR", "ne": "Normal\nequations"}
STYLE = {"naive": ("k", "o", "-"), "qr": ("C0", "s", "-"), "qr-tree": ("C2", "^", "-"),
         "ne": ("C3", "v", "--"), "ne-reset": ("C1", "D", "--"), "qr-seq": ("C4", "x", "--")}

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.2,
                     "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight"})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def fig1(s, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for mth in ["naive", "qr", "qr-tree", "ne", "ne-reset"]:
        g = s[s.method == mth].groupby("W").t_min
        if g.ngroups == 0:
            continue
        W = np.array(list(g.groups))
        med, lo, hi = g.median().values, g.min().values, g.max().values
        c, mk, ls = STYLE[mth]
        ax.plot(W, med, color=c, marker=mk, ls=ls, label=LABEL[mth])
        ax.fill_between(W, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Window size W (rows)")
    ax.set_ylabel("Total time (s)")
    N = int(s.N.iloc[0])
    ax.set_title(f"Sliding windows, N = {N:.0e}, b = {int(s.b.iloc[0])}", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    save(fig, out, "fig1_tiempo_vs_W")


def fig2(s, out):
    Ws = sorted(s.W.unique())
    Wsel = [Ws[0], Ws[-1]] if len(Ws) > 1 else Ws
    fig, axs = plt.subplots(1, len(Wsel), figsize=(3.5 * len(Wsel), 2.8), sharey=True, squeeze=False)
    for ax, W in zip(axs[0], Wsel):
        for mth in ["qr", "qr-tree", "ne", "ne-reset"]:
            g = s[(s.method == mth) & (s.W == W)]
            if g.empty:
                continue
            c, mk, ls = STYLE[mth]
            ax.scatter(g.kappa_median, g.err_median, color=c, marker=mk, s=10, alpha=0.6)
            agg = g.groupby("level").agg(k=("kappa_median", "median"), e=("err_median", "median")).sort_values("k")
            ax.plot(agg.k, agg.e, color=c, ls=ls, label=LABEL[mth])
        k = np.logspace(np.log10(s.kappa_median.min()) - 0.2, np.log10(s.kappa_median.max()) + 0.2, 50)
        ax.plot(k, k * np.finfo(float).eps, color="0.5", ls=":", lw=1, label="κ·u")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("κ(X) of the window (median)")
        ax.set_title(f"W = {W}", fontsize=8.5)
    axs[0, 0].set_ylabel("Relative error w.r.t. lstsq (median)")
    axs[0, 0].legend(loc="upper left", frameon=False)
    save(fig, out, "fig2_error_vs_kappa")


def fig3(s, out):
    """Error normalizado por κ·ε para separar el efecto de m del cambio de κ con W."""
    levels = sorted(s.level.unique())
    s = s.assign(err_norm=s.err_median / (s.kappa_median * np.finfo(float).eps))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    cmap = plt.get_cmap("viridis", len(levels) + 1)
    for i, lv in enumerate(levels):
        for mth, ls, mk in (("qr", "-", "s"), ("qr-tree", "--", "^")):
            g = s[(s.method == mth) & (s.level == lv)].groupby("m").err_norm.median()
            if g.empty:
                continue
            kap = s[s.level == lv].kappa_median.median()
            ax.plot(g.index, g.values, color=cmap(i), ls=ls, marker=mk,
                    label=f"{'two-stacks' if mth == 'qr' else 'tree'}, κ≈{kap:.0e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Blocks per window m = W/b")
    ax.set_ylabel("Error / (κ·u) (median)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.5)
    save(fig, out, "fig3_error_vs_m")


def fig4(e, out):
    if e.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for mth in ["qr", "qr-seq", "ne"]:
        g = e[e.method == mth]
        if g.empty:
            continue
        c, mk, ls = STYLE[mth]
        agg = g.groupby("level").agg(k=("kappa_median", "median"), e=("err_median", "median"),
                                     lo=("err_median", "min"), hi=("err_max", "max")).sort_values("k")
        a1.plot(agg.k, agg.e, color=c, marker=mk, ls=ls, label=LABEL_EXP[mth] + " (median)")
        a1.fill_between(agg.k, agg.lo, agg.hi, color=c, alpha=0.15, lw=0)
    k = np.logspace(np.log10(e.kappa_median.min()) - 0.2, np.log10(e.kappa_median.max()) + 0.2, 50)
    a1.plot(k, k * np.finfo(float).eps, color="0.5", ls=":", lw=1, label="κ·u")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("κ(X) of the prefix (median)")
    a1.set_ylabel("Relative error w.r.t. lstsq")
    a1.set_title(f"Expanding windows, N = {int(e.N.iloc[0]):.0e}", fontsize=8.5)
    a1.legend(loc="upper left", frameon=False, fontsize=6.5)
    meths = [m for m in ["qr", "qr-seq", "ne"] if m in set(e.method)]
    t = [e[e.method == m].t_min for m in meths]
    a2.boxplot(t, tick_labels=[SHORT_EXP[m] for m in meths], widths=0.5)
    a2.set_ylabel("Total time (s)")
    a2.set_title("Time (all seeds and levels)", fontsize=8.5)
    save(fig, out, "fig4_expanding")


def tabla(d, out):
    t = (d.groupby(["mode", "method", "W", "level"])
         .agg(kappa=("kappa_median", "median"), t_min=("t_min", "median"),
              us_ventana=("t_per_window_us", "median"), err_mediana=("err_median", "median"),
              err_max=("err_max", "max"), semillas=("seed", "nunique"))
         .reset_index())
    t.to_csv(os.path.join(out, "tabla_resumen.csv"), index=False)
    print("  tabla_resumen.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figuras")
    ap.add_argument("--exclude-levels", type=float, nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.csv)
    if a.exclude_levels:
        d = d[~d.level.isin(a.exclude_levels)]
    # Comprobación de la referencia: con rcond=None, lstsq truncaba el rango y los errores de
    # todos los métodos QR salían O(1) a la vez. Se avisa en lugar de graficar en silencio.
    bad = d[(d.method.isin(["qr", "qr-tree"])) & (d.err_median > 1e-3)]
    if len(bad):
        print("AVISO: errores QR > 1e-3 en", sorted(set(zip(bad["mode"], bad.level, bad.W))),
              "\n       probablemente la referencia lstsq truncó el rango (versión con rcond=None).")
    s, e = d[d["mode"] == "sliding"], d[d["mode"] == "expanding"]
    print(f"filas: {len(d)}  (sliding {len(s)}, expanding {len(e)})  niveles: {sorted(d.level.unique())}")
    if not s.empty:
        fig1(s, a.out)
        fig2(s[s.method != "naive"], a.out)
        fig3(s[s.method != "naive"], a.out)
    fig4(e, a.out)
    tabla(d, a.out)
    print(f"-> {a.out}/")


if __name__ == "__main__":
    main()
