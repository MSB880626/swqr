# swqr: downdating-free sliding- and expanding-window least squares

Code and benchmark data for the article

> M. Sánchez Brito, *Downdating-free sliding- and expanding-window least squares by associative aggregation of QR factors* (submitted to the Journal of Computational and Applied Mathematics).

Rolling (sliding-window) and expanding-window linear regressions are solved by aggregating triangular QR factors of the augmented matrix `[X y]` with the associative, non-invertible merge `R1 ⊕ R2 = R([R1; R2])`. No downdating is ever performed.

| Method (`--method`) | Window | Structure | Merges per window | Error growth |
|---|---|---|---|---|
| `qr` | sliding | two-stacks queue | O(1) amortized | O(W/b) |
| `qr-tree` | sliding | circular segment tree | O(log(W/b)) | O(log(W/b)) |
| `qr` | expanding | binary counter | O(log k) | O(log k) |
| `qr-seq` | expanding | sequential QR updating (baseline) | 1 | O(k) |
| `ne`, `ne-reset` | both | normal equations with update/downdate (baselines) | — | ∝ κ² |
| `givens`, `naive` | sliding | Givens update + LINPACK downdate; `lstsq` per window (baselines) | — | — |

## Contents

| Path | Description |
|---|---|
| `swqr_spark.py` | Algorithms (NumPy and Numba engines), data generator, accuracy checks, benchmark and distributed PySpark implementation |
| `compare_software.py` | Comparison with statsmodels RollingOLS and the R package rollRegres |
| `test_swqr_spark.py` | Unit and integration tests (including Spark end-to-end) |
| `figures/` | Scripts that produce the figures and tables of the article from the CSV files |
| `results/` | Raw benchmark results used in the article |
| `scripts/install_rollregres.sh` | Installs rollRegres 0.1.4 (archived on CRAN) with a compatibility patch for R ≥ 4.5 |

## Installation

```bash
pip install -r requirements.txt
# Java 17 is required only for the PySpark implementation
# R and rollRegres are required only for compare_software.py:
bash scripts/install_rollregres.sh
```

## Tests

```bash
python test_swqr_spark.py                    # ~3 min including Spark
SWQR_SKIP_SPARK=1 python test_swqr_spark.py   # without Spark
```

## Reproducing the results of the article

```bash
# Sliding and expanding benchmarks, N = 1e6 (results/bench_1e6.csv)
python swqr_spark.py bench --csv bench_1e6.csv --n 1000000 --resume

# Validation against a 60-digit reference (results/bench_mp.csv)
python swqr_spark.py bench --csv bench_mp.csv --n 60000 --windows 1000 10000 \
    --levels 1e2 1e3 1e4 1e5 --seeds 1 2 3 --mp-points 6

# Comparison with statsmodels and rollRegres (results/comparison_b1.csv, results/comparison_b10.csv)
python compare_software.py --csv comparison_b1.csv  --n 1000000 --mp-points 0 --resume
python compare_software.py --csv comparison_b10.csv --n 1000000 --block 10 --mp-points 0 --sm-lstsq-max 0 --resume

# Figures and summary tables
python figures/figures_bench.py      --csv results/bench_1e6.csv     --out out_figures
python figures/figures_comparison.py --csv results/comparison_b1.csv --out out_figures_comparison
```

Correspondence between the article and the files: Fig. 1 = `fig2_error_vs_kappa`, Fig. 2 = `fig3_error_vs_m`, Fig. 3 = `fig4_expanding`, Fig. 4 = `fig1_tiempo_vs_W` (from `figures_bench.py`); Figs. 5 and 6 = `fig5_comparacion_error` and `fig6_comparacion_tiempo` (from `figures_comparison.py`).

## Other commands

```bash
python swqr_spark.py gen    --out data --n 60000                          # synthetic ARX series (Parquet)
python swqr_spark.py check  --data data --mode sliding --window 1000       # accuracy and time vs lstsq
python swqr_spark.py verify --data data --mode expanding                   # accuracy vs 60-digit reference
python swqr_spark.py run    --data data --mode sliding --method qr \
       --window 1000 --block 10 --partitions 7 --master "local[*]" --out res   # distributed (PySpark)
```

Use `python swqr_spark.py <command> --help` for all options.

## Notes

- The least squares reference uses `numpy.linalg.lstsq(..., rcond=-1)`. The default cutoff (`rcond=None`, i.e. ε·max(m, n)) truncates genuine singular values of tall ill-conditioned matrices (κ > ~4.5e9 with 10⁶ rows).
- Source comments are partly in Spanish.

## License

MIT (see `LICENSE`). If you use this code, please cite the article and the Zenodo record (see `CITATION.cff`).
