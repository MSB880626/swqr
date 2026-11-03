#!/usr/bin/env python3
"""
figuras_comparacion.py — figuras y tabla de la comparación contra software existente
(salida de compare_software.py).

Uso:
  python figuras_comparacion.py --csv comparacion_1e6.csv --out figuras_comp

Genera (PNG 300 dpi y PDF):
  fig5_comparacion_error   error relativo mediano (barra: máximo) vs κ, sliding por W y expanding
  fig6_comparacion_tiempo  tiempo total por método, sliding por W y expanding
  tabla_comparacion.csv    tiempo, error mediano y máximo por método, W y nivel
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = {("swqr", "qr"): ("This work: QR two-stacks / binary counter", "C0", "s", "-"),
       ("swqr", "qr-tree"): ("This work: QR segment tree", "C2", "^", "-"),
       ("swqr", "qr-seq"): ("This work: sequential QR", "C4", "x", "-"),
       ("swqr", "ne"): ("This work: normal equations", "C3", "v", ":"),
       ("rollRegres", "qr-updown"): ("rollRegres: QR update/downdate", "k", "o", "--"),
       ("rollRegres", "qr-update"): ("rollRegres: QR update", "k", "o", "--"),
       ("statsmodels", "inv"): ("statsmodels RollingOLS (inv)", "C1", "D", "--"),
       ("statsmodels", "inv-reset"): ("statsmodels RollingOLS (inv, reset)", "C5", "P", "--"),
       ("statsmodels", "lstsq"): ("statsmodels RollingOLS (lstsq)", "C7", "*", "--")}

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.2,
                     "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight"})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def panels(d):
    out = [("sliding", W) for W in sorted(d[d["mode"] == "sliding"].W.unique())]
    if (d["mode"] == "expanding").any():
        out.append(("expanding", int(d[d["mode"] == "expanding"].W.iloc[0])))
    return out


def fig5(d, out):
    P = panels(d)
    fig, axs = plt.subplots(1, len(P), figsize=(2.6 * len(P), 2.9), sharey=True, squeeze=False)
    handles = {}
    for ax, (mode, W) in zip(axs[0], P):
        g = d[(d["mode"] == mode) & (d.W == W) & (d.method != "lstsq")]
        for (sw, mth), h in g.groupby(["software", "method"]):
            if (sw, mth) not in KEY:
                continue
            lab, c, mk, ls = KEY[(sw, mth)]
            h = h.groupby("level").agg(k=("kappa_median", "median"), e=("err_median", "median"),
                                       mx=("err_max", "max")).sort_values("k")
            ln, = ax.plot(h.k, h.e, color=c, marker=mk, ls=ls)
            ax.vlines(h.k, h.e, h.mx, color=c, alpha=0.35, lw=2)
            handles[lab] = ln
        k = np.logspace(np.log10(d.kappa_median.min()) - 0.2, np.log10(d.kappa_median.max()) + 0.2, 30)
        ax.plot(k, k * np.finfo(float).eps, color="0.5", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("κ(X) (median)")
        ax.set_title(f"{'Sliding' if mode == 'sliding' else 'Expanding'}, W = {W}", fontsize=8.5)
    axs[0, 0].set_ylabel("Relative error w.r.t. lstsq\n(median; bar: maximum)")
    order = [KEY[k][0] for k in KEY if KEY[k][0] in handles]
    fig.legend([handles[l] for l in order], order, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    b = int(d.b.iloc[0]) if "b" in d.columns else 1
    fig.suptitle(f"N = {int(d.N.iloc[0]):.0e}, b = {b}; dotted line: κ·u", fontsize=8.5, y=1.02)
    save(fig, out, "fig5_comparacion_error")


def fig6(d, out):
    P = panels(d)
    fig, axs = plt.subplots(1, len(P), figsize=(2.6 * len(P), 3.0), sharey=True, squeeze=False)
    for ax, (mode, W) in zip(axs[0], P):
        g = d[(d["mode"] == mode) & (d.W == W)].groupby(["software", "method"]).time_s.median()
        keys = [k for k in KEY if k in g.index]
        x = np.arange(len(keys))
        ax.bar(x, [g[k] for k in keys], color=[KEY[k][1] for k in keys])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], rotation=90, fontsize=6.5)
        ax.set_yscale("log")
        ax.set_title(f"{'Sliding' if mode == 'sliding' else 'Expanding'}, W = {W}", fontsize=8.5)
    axs[0, 0].set_ylabel("Total time (s), median over κ")
    save(fig, out, "fig6_comparacion_tiempo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figuras_comp")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.csv)
    if "b" not in d.columns:
        d["b"] = 1
    for b, g in d.groupby("b"):
        sub = os.path.join(a.out, f"b{int(b)}")
        os.makedirs(sub, exist_ok=True)
        print(f"b = {int(b)}: {len(g)} filas -> {sub}/")
        fig5(g, sub)
        fig6(g, sub)
    t = (d.groupby(["b", "mode", "W", "level", "software", "method"])
         .agg(kappa=("kappa_median", "median"), time_s=("time_s", "median"),
              us_ventana=("us_per_window", "median"), err_mediana=("err_median", "median"),
              err_max=("err_max", "max")).reset_index())
    t.to_csv(os.path.join(a.out, "tabla_comparacion.csv"), index=False)
    print(f"  tabla_comparacion.csv -> {a.out}/")


if __name__ == "__main__":
    main()
