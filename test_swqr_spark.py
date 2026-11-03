"""Pruebas de swqr_spark.py.

- Corrección de todos los métodos frente a lstsq con datos bien condicionados.
- Lógica distribuida real (partition_summary / plan_partitions / partition_compute) con
  cortes de partición no alineados, sin requerir Spark.
- Exactitud con datos mal condicionados (κ ~ 1e8-1e9) frente a una referencia mpmath:
  es la afirmación central del artículo.
- Extremo a extremo con Spark (`run`), solo si pyspark y Java están disponibles.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("swqr_spark", HERE / "swqr_spark.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

HAS_MPMATH = importlib.util.find_spec("mpmath") is not None
ENGINES = ("numpy", "numba") if M.HAS_NUMBA else ("numpy",)
HAS_SPARK = (importlib.util.find_spec("pyspark") is not None and shutil.which("java") is not None
             and not os.environ.get("SWQR_SKIP_SPARK"))


def well_conditioned(n=173, seed=2026):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.2, -0.8, 0.3, 2.1]) + rng.normal(0, .2, n)
    return M.design(X, y)


def ill_conditioned(n, seed=12345):
    _, chunks = M.arx_chunks(n, seed=seed, chunk=max(n, 1000))
    t, y, X = (np.concatenate(v) for v in zip(*chunks))
    return M.design(X, y)


def run_distributed(A, cuts, W, b, mode, method, reset=0, engine="numpy"):
    """Ejecuta el mismo camino de código que cmd_run, partición por partición, sin Spark."""
    parts, summaries = [], []
    for pid, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        t = np.arange(lo, hi, dtype=np.int64)
        parts.append((pid, t, A[lo:hi]))
        s = M.partition_summary(t, A[lo:hi], W, b, mode, method)
        s["pid"] = pid
        summaries.append(s)
    info, _ = M.plan_partitions(summaries, W, b, mode, method)
    out = []
    for pid, t, Ap in parts:
        out.extend(M.partition_compute(t, Ap, info[pid], W, b, mode, method, reset, engine))
    return out


class TestCorrectness(unittest.TestCase):
    def setUp(self):
        self.A = well_conditioned()

    def test_sliding_methods_against_lstsq(self):
        W, b = 30, 5
        ref = M.sliding_windows(self.A, 0, b, W, "naive")
        for engine in ENGINES:
            for method in ("qr", "qr-tree", "ne", "ne-reset", "givens"):
                got = M.sliding_windows(self.A, 0, b, W, method, reset=7, engine=engine)
                self.assertEqual([r[0] for r in got], [r[0] for r in ref], (engine, method))
                for g, z in zip(got, ref):
                    np.testing.assert_allclose(g[1], z[1], rtol=1e-8, atol=1e-9)
                    self.assertAlmostEqual(g[2], z[2], delta=1e-8)

    def test_expanding_methods_against_lstsq(self):
        for engine in ENGINES:
            for method in ("qr", "qr-seq", "ne"):
                for e, beta, rss in M.expanding_windows(self.A, 0, 5, method, min_rows=20, engine=engine):
                    bref, rref = M.solve_lstsq(self.A[:e + 1])
                    np.testing.assert_allclose(beta, bref, rtol=1e-8, atol=1e-9)
                    self.assertAlmostEqual(rss, rref, delta=1e-7)

    @unittest.skipUnless(M.HAS_NUMBA, "requiere numba")
    def test_numba_kernels_match_numpy(self):
        rng = np.random.default_rng(7)
        for n, m in ((5, 3), (5, 11), (12, 40)):
            B = rng.standard_normal((m, n))
            np.testing.assert_allclose(M._nb_leaf(B), M.r_factor(B), atol=1e-12)
            R1, R2 = M.r_factor(rng.standard_normal((20, n))), M.r_factor(rng.standard_normal((20, n)))
            np.testing.assert_allclose(M._nb_combine(R1, R2), M.combine(R1, R2), atol=1e-12)

    def test_tree_window_sizes(self):
        # m = W/b potencia de 2 (sin envoltura parcial), m < M (consultas que envuelven) y m = 1
        for W, b in ((40, 5), (30, 5), (45, 5), (5, 5)):
            ref = M.sliding_windows(self.A, 0, b, W, "naive")
            for engine in ENGINES:
                got = M.sliding_windows(self.A, 0, b, W, "qr-tree", engine=engine)
                self.assertEqual([r[0] for r in got], [r[0] for r in ref], (W, b, engine))
                for g, z in zip(got, ref):
                    np.testing.assert_allclose(g[1], z[1], rtol=1e-8, atol=1e-9)

    def test_binary_counter_equals_full_qr(self):
        bc = M.BinaryCounterR()
        for k in range(0, 170, 10):
            bc.push(M.r_factor(self.A[k:k + 10]))
            np.testing.assert_allclose(bc.fold(), M.r_factor(self.A[:k + 10]), atol=1e-12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            M.sliding_windows(self.A, 0, 5, 0)
        with self.assertRaises(ValueError):
            M.sliding_windows(self.A, 0, 5, 12)
        with self.assertRaises(ValueError):
            M.sliding_windows(self.A, 3, 5, 30)
        with self.assertRaises(ValueError):
            M.expanding_windows(self.A, 0, 5, "naive")
        with self.assertRaises(ValueError):
            M.sliding_windows(self.A, 0, 5, 30, "qr-seq")


class TestDistributedLogic(unittest.TestCase):
    """Cortes no alineados a b, una partición de 1 fila y particiones menores que W+b."""

    CUTS = [0, 17, 43, 44, 91, 128, 173]

    def setUp(self):
        self.A = well_conditioned()

    def test_sliding(self):
        W, b = 30, 5
        ref = M.sliding_windows(self.A, 0, b, W, "naive")
        for engine in ENGINES:
            for method in M.DIST_SLIDING_METHODS:
                got = run_distributed(self.A, self.CUTS, W, b, "sliding", method, reset=3, engine=engine)
                self.assertEqual([r[0] for r in got], [r[0] for r in ref], (engine, method))
                for g, z in zip(got, ref):
                    np.testing.assert_allclose(g[1], z[1], rtol=1e-8, atol=1e-9)

    def test_expanding(self):
        W, b = 30, 5
        for engine in ENGINES:
            for method in M.DIST_EXPANDING_METHODS:
                got = run_distributed(self.A, self.CUTS, W, b, "expanding", method, engine=engine)
                ref = M.expanding_windows(self.A, 0, b, method, min_rows=W)
                self.assertEqual([r[0] for r in got], [r[0] for r in ref], (engine, method))
                for g, (e, _, _) in zip(got, ref):
                    np.testing.assert_allclose(g[1], M.solve_lstsq(self.A[:e + 1])[0], rtol=1e-8, atol=1e-9)

    def test_plan_rejects_gap(self):
        W, b = 30, 5
        s1 = M.partition_summary(np.arange(0, 50), self.A[:50], W, b, "sliding", "qr")
        s2 = M.partition_summary(np.arange(60, 100), self.A[60:100], W, b, "sliding", "qr")
        s1["pid"], s2["pid"] = 0, 1
        with self.assertRaises(RuntimeError):
            M.plan_partitions([s1, s2], W, b, "sliding", "qr")


@unittest.skipUnless(HAS_MPMATH, "requiere mpmath")
class TestIllConditioned(unittest.TestCase):
    """κ(X) ~ 1e8-1e9. Referencia en 60 dígitos; lstsq es exacta a ~1e-10 aquí."""

    @classmethod
    def setUpClass(cls):
        cls.A = ill_conditioned(30000)

    def _errors(self, mode, methods, W, b, npts, engine):
        ends = [e for e in M.aligned_ends(0, len(self.A) - 1, b) if e + 1 >= W]
        pts = sorted(set(ends[:: max(1, len(ends) // npts)]) | {ends[-1]})
        ref = M.reference_mp(self.A, pts, W, mode)
        errs = {}
        for m in methods:
            out = (M.sliding_windows(self.A, 0, b, W, m, reset=100, engine=engine) if mode == "sliding"
                   else M.expanding_windows(self.A, 0, b, m, min_rows=W, engine=engine))
            d = {e: beta for e, beta, _ in out}
            errs[m] = max(M.rel_err(d[e], ref[e])[0] for e in pts)
        return errs

    def test_conditioning_is_high(self):
        self.assertGreater(np.linalg.cond(self.A[:1000, :-1]), 1e8)

    def test_sliding_qr_vs_normal_equations(self):
        for engine in ENGINES:
            e = self._errors("sliding", ("qr", "qr-tree", "ne", "ne-reset"), 1000, 10, 8, engine)
            for m in ("qr", "qr-tree"):
                self.assertLess(e[m], 1e-6, (engine, m))
                self.assertGreater(e["ne"], 100 * e[m], (engine, m))
                self.assertGreater(e["ne-reset"], 100 * e[m], (engine, m))

    def test_expanding_tree_vs_sequential_vs_ne(self):
        for engine in ENGINES:
            e = self._errors("expanding", ("qr", "qr-seq", "ne"), 1000, 10, 10, engine)
            self.assertLess(e["qr"], 5e-8, engine)
            self.assertLess(e["qr"], e["qr-seq"], engine)
            self.assertGreater(e["ne"], 100 * e["qr"], engine)


class TestBench(unittest.TestCase):
    def test_bench_csv_and_resume(self):
        import csv

        script = str(HERE / "swqr_spark.py")
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "b.csv")
            args = [sys.executable, script, "bench", "--csv", out, "--n", "6000", "--windows", "100", "200",
                    "--levels", "1e3", "--seeds", "1", "--repeats", "1", "--err-points", "10",
                    "--exp-min-rows", "100"] + (["--mp-points", "2"] if HAS_MPMATH else [])
            subprocess.run(args, check=True, capture_output=True)
            with open(out) as f:
                rows = list(csv.DictReader(f))
            # sliding: 2 W x (naive + 4 métodos); expanding: 3 métodos
            self.assertEqual(len(rows), 2 * 5 + 3)
            self.assertEqual(set(rows[0]), set(M.BENCH_FIELDS))
            for r in rows:
                if r["method"] != "naive":
                    self.assertLess(float(r["err_max"]), 1e-3, r)
            r2 = subprocess.run(args + ["--resume"], check=True, capture_output=True, text=True)
            self.assertIn("reanudando: 13", r2.stdout)
            with open(out) as f:
                self.assertEqual(len(list(csv.DictReader(f))), 13)


@unittest.skipUnless(HAS_SPARK, "requiere pyspark y java")
class TestSparkEndToEnd(unittest.TestCase):
    def test_run_matches_single_node(self):
        script = str(HERE / "swqr_spark.py")
        with tempfile.TemporaryDirectory() as d:
            data = os.path.join(d, "datos")
            subprocess.run([sys.executable, script, "gen", "--out", data, "--n", "3000"],
                           check=True, capture_output=True)
            _, A = M.load_numpy(data, True)
            for mode, method, engine in (("sliding", "qr", "auto"), ("sliding", "qr-tree", "auto"),
                                         ("expanding", "qr", "auto"),
                                         ("expanding", "qr-seq", "numpy")):
                out = os.path.join(d, f"res_{mode}_{method}")
                subprocess.run([sys.executable, script, "run", "--data", data, "--mode", mode,
                                "--method", method, "--window", "100", "--block", "10",
                                "--partitions", "7", "--master", "local[2]", "--engine", engine,
                                "--out", out], check=True, capture_output=True)
                import pandas as pd

                got = pd.read_parquet(out).sort_values("t_end")
                ref = (M.sliding_windows(A, 0, 10, 100, "naive") if mode == "sliding"
                       else [(e,) + M.solve_lstsq(A[:e + 1]) for e in
                             M.aligned_ends(0, len(A) - 1, 10) if e + 1 >= 100])
                self.assertEqual(list(got.t_end), [r[0] for r in ref], (mode, method))
                err = M.rel_err(np.vstack(got.beta.to_numpy()), np.vstack([r[1] for r in ref]))
                self.assertLess(err.max(), 1e-6, (mode, method))


if __name__ == "__main__":
    unittest.main(verbosity=2)
