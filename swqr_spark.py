#!/usr/bin/env python3
"""
swqr_spark.py
Regresión lineal múltiple en ventana deslizante y expansiva sobre PySpark sin downdating.

Operador: R(A1 ⊕ A2) = R(QR([R1; R2])) sobre la matriz aumentada A = [1 | X | y],
asociativo y no invertible.
  - sliding:   two-stacks sobre factores R de bloque; O(W/b) combinaciones por ventana,
               error independiente de N. Halo de a lo sumo W+b-1 filas por frontera.
  - expanding: contador binario (reducción en árbol tipo TSQR) dentro de cada partición
               y entre particiones; cada prefijo es la combinación de O(log k) subárboles
               balanceados, de modo que el error de redondeo crece como O(log k) y no O(k).

Métodos (--method / --methods):
  qr        sliding: two-stacks, O(1) combinaciones amortizadas por desplazamiento, error O(W/b)
            expanding: contador binario, error O(log k)
  qr-tree   solo sliding: árbol de segmentos circular sobre los bloques de la ventana,
            O(log(W/b)) combinaciones por desplazamiento, error O(log(W/b))
  qr-seq    solo expanding: prefijo QR secuencial (una cadena de k combinaciones)
  ne        ecuaciones normales con update/downdate por bloque (expanding: suma acumulada)
  ne-reset  solo sliding: ecuaciones normales con recálculo de G cada --reset ventanas
  givens    solo sliding y en un nodo: update de Givens + downdating de Saunders por fila
  naive     solo sliding: lstsq (SVD, LAPACK gelsd) sobre cada ventana

Subcomandos:
  gen     serie ARX sintética mal condicionada -> Parquet (t, y, x0..x{p-1})
  check   exactitud y tiempo en un nodo contra lstsq en float64
  verify  exactitud contra una referencia en alta precisión (mpmath), opcionalmente
          también de una salida de `run`
  run     ejecución distribuida (spark-submit, local[*] o standalone)
  bench   barrido de tiempo y exactitud (N, W, b, nivel/κ, semilla) -> CSV

Motores (--engine): numba (kernels compilados; por omisión si está instalado) o numpy.
Requisitos: numpy, scipy, pandas, pyarrow; numba (recomendado); pyspark para `run`;
mpmath para `verify`.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
from scipy.linalg import solve_triangular

SLIDING_METHODS = ("qr", "qr-tree", "ne", "ne-reset", "givens", "naive")
EXPANDING_METHODS = ("qr", "qr-seq", "ne")
DIST_SLIDING_METHODS = ("qr", "qr-tree", "ne", "ne-reset", "naive")
DIST_EXPANDING_METHODS = ("qr", "qr-seq", "ne")

# ----------------------------------------------------------------------------- núcleo


def r_factor(A):
    """Factor R (n x n) de Householder con diagonal no negativa (único si A es de rango completo)."""
    n = A.shape[1]
    R = np.linalg.qr(A, mode="r")
    if R.shape[0] < n:
        R = np.vstack([R, np.zeros((n - R.shape[0], n))])
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return d[:, None] * R


def combine(R1, R2):
    if R1 is None:
        return R2
    if R2 is None:
        return R1
    return r_factor(np.vstack([R1, R2]))


def solve_R(R):
    p = R.shape[1] - 1
    Rxx, rxy, rho = R[:p, :p], R[:p, p], R[p, p]
    if np.any(np.abs(np.diag(Rxx)) == 0.0):
        return np.full(p, np.nan), np.nan
    return solve_triangular(Rxx, rxy, lower=False, check_finite=False), float(rho * rho)


def solve_G(G):
    p = G.shape[1] - 1
    Gxx, gxy = G[:p, :p], G[:p, p]
    try:
        beta = np.linalg.solve(Gxx, gxy)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), np.nan
    return beta, float(G[p, p] - gxy @ beta)


def solve_lstsq(Wd):
    # rcond=-1: corte en ε de máquina. Con rcond=None numpy corta en ε·max(m, n), que con
    # m = 10^6 filas descarta valores singulares reales cuando κ > ~4.5e9 y devuelve la
    # solución de norma mínima de un problema de rango reducido (no la de mínimos cuadrados).
    beta, _, _, _ = np.linalg.lstsq(Wd[:, :-1], Wd[:, -1], rcond=-1)
    r = Wd[:, -1] - Wd[:, :-1] @ beta
    return beta, float(r @ r)


class TwoStacksR:
    """Agregación FIFO de factores R sin inversa: push/evict O(1) combinaciones amortizadas."""

    def __init__(self):
        self.back = []
        self.back_agg = None
        self.front = []  # front[-1] = agregado del elemento más antiguo hasta el final del front

    def __len__(self):
        return len(self.front) + len(self.back)

    def push(self, R):
        self.back.append(R)
        self.back_agg = combine(self.back_agg, R)

    def evict(self):
        if not self.front:
            acc = None
            for R in reversed(self.back):
                acc = combine(R, acc)
                self.front.append(acc)
            self.back, self.back_agg = [], None
        self.front.pop()

    def query(self):
        return combine(self.front[-1] if self.front else None, self.back_agg)


class RingSegTreeR:
    """Árbol de segmentos circular de M = 2^ceil(log2 m) hojas con factores R de bloque.

    set(i, R): O(log M) combinaciones. query(first, last): O(log M) nodos, cada uno una
    reducción balanceada de sus hojas; ninguna hoja pasa por más de ~3 log2 M combinaciones.
    """

    def __init__(self, m):
        M = 1
        while M < m:
            M *= 2
        self.M = M
        self.T = [None] * (2 * M)

    def set(self, i, R):
        j = self.M + (i % self.M)
        self.T[j] = R
        j //= 2
        while j >= 1:
            self.T[j] = combine(self.T[2 * j], self.T[2 * j + 1])
            j //= 2

    def _range(self, lo, hi, acc):
        lo += self.M
        hi += self.M + 1
        while lo < hi:
            if lo & 1:
                acc = combine(acc, self.T[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                acc = combine(acc, self.T[hi])
            lo //= 2
            hi //= 2
        return acc

    def query(self, first, last):
        """Agregado de las hojas globales first..last (last - first + 1 <= M)."""
        if last - first + 1 == self.M:
            return self.T[1]
        lo, hi = first % self.M, last % self.M
        if lo <= hi:
            return self._range(lo, hi, None)
        return self._range(0, hi, self._range(lo, self.M - 1, None))


class BinaryCounterR:
    """Prefijo de factores R como contador binario: subárboles balanceados de tamaño 2^l.

    push: O(1) combinaciones amortizadas. fold: O(log k) combinaciones.
    Cada hoja pasa por O(log k) combinaciones desde que entra hasta el prefijo consultado.
    """

    def __init__(self):
        self.stack = []  # (nivel, R); niveles estrictamente decrecientes hacia el tope

    def push(self, R):
        lvl = 0
        while self.stack and self.stack[-1][0] == lvl:
            _, Rp = self.stack.pop()
            R = combine(Rp, R)
            lvl += 1
        self.stack.append((lvl, R))

    def fold(self):
        acc = None
        for _, R in reversed(self.stack):  # del subárbol más pequeño al más grande
            acc = combine(R, acc)
        return acc


# ------------------------------------------------- QR con update/downdate por fila (línea base)


def _givens_update(R, x):
    """Agrega la fila x a R (n x n) con rotaciones de Givens. O(n^2)."""
    R = R.copy()
    x = x.astype(float).copy()
    n = R.shape[1]
    for k in range(n):
        a, b = R[k, k], x[k]
        if b == 0.0:
            continue
        r = np.hypot(a, b)
        c, s = a / r, b / r
        Rk = R[k, k:].copy()
        R[k, k:] = c * Rk + s * x[k:]
        x[k:] = -s * Rk + c * x[k:]
    return R


def _downdate(R, x):
    """Elimina la fila x de R (algoritmo de Saunders / LINPACK dchdd). O(n^2). Inestable si mal condicionado."""
    n = R.shape[1]
    a = solve_triangular(R, x, trans="T", lower=False, check_finite=False)
    alpha2 = 1.0 - a @ a
    if alpha2 <= 0.0:
        raise np.linalg.LinAlgError("downdate no definido (pérdida de definitud)")
    alpha = np.sqrt(alpha2)
    c = np.empty(n)
    s = np.empty(n)
    for i in range(n - 1, -1, -1):
        scale = alpha + abs(a[i])
        aa, bb = alpha / scale, a[i] / scale
        norm = scale * np.hypot(aa, bb)
        c[i], s[i] = alpha / norm, a[i] / norm
        alpha = norm
    Rn = R.copy()
    for j in range(n):
        xx = 0.0
        for i in range(j, -1, -1):
            t = c[i] * xx + s[i] * Rn[i, j]
            Rn[i, j] = c[i] * Rn[i, j] - s[i] * xx
            xx = t
    d = np.sign(np.diag(Rn))
    d[d == 0] = 1.0
    return d[:, None] * Rn


# ------------------------------------------------------------- kernels Numba
#
# Mismo algoritmo que la versión numpy, compilado. La combinación R1 ⊕ R2 explota la
# estructura triangular de ambos factores: en la columna k solo participan R1[k,k] y las
# filas 0..k de R2, de modo que cuesta ~ (2/3) n^3 flops en lugar de ~ (10/3) n^3 de un
# Householder denso sobre [R1; R2]. La hoja R(bloque) y la actualización secuencial (qr-seq)
# usan la misma rutina con todas las filas del bloque activas.
# fastmath={'contract'} solo permite contraer a*b+c en FMA (sin reordenar sumas ni
# suponer ausencia de NaN/Inf); da la misma exactitud que LAPACK/OpenBLAS.

try:
    import numba as _numba

    HAS_NUMBA = True
    _jit = _numba.njit(cache=True, fastmath={'contract'})
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def _jit(f):
        return f

NUMBA_SLIDING = ("qr", "qr-tree", "ne", "ne-reset")
NUMBA_EXPANDING = ("qr", "qr-seq", "ne")


@_jit
def _nb_update(R, B, tri):
    """R <- R( [R; B] ) con reflectores de Householder en la forma de LAPACK (dlarfg/dlarf).
    R (n x n) triangular superior se actualiza en el lugar; B (mb x n) se destruye.
    tri=True: B es triangular superior y en la columna k solo están activas sus filas 0..k."""
    n = R.shape[1]
    mb = B.shape[0]
    v = np.empty(mb)
    for k in range(n):
        top = min(k + 1, mb) if tri else mb
        x0 = R[k, k]
        s = 0.0
        for i in range(top):
            s += B[i, k] * B[i, k]
        if s == 0.0:
            continue
        beta = -np.copysign(np.hypot(x0, np.sqrt(s)), x0) if x0 != 0.0 else -np.sqrt(s)
        tau = (beta - x0) / beta
        sc = 1.0 / (x0 - beta)
        for i in range(top):
            v[i] = B[i, k] * sc
        for j in range(k + 1, n):
            d = R[k, j]
            for i in range(top):
                d += v[i] * B[i, j]
            d *= tau
            R[k, j] -= d
            for i in range(top):
                B[i, j] -= d * v[i]
        R[k, k] = beta
        for i in range(top):
            B[i, k] = 0.0


@_jit
def _nb_signfix(R):
    n = R.shape[0]
    for i in range(n):
        if R[i, i] < 0.0:
            for j in range(i, n):
                R[i, j] = -R[i, j]


@_jit
def _nb_leaf(blk):
    n = blk.shape[1]
    R = np.zeros((n, n))
    B = blk.copy()
    _nb_update(R, B, False)
    _nb_signfix(R)
    return R


@_jit
def _nb_combine(R1, R2):
    R = R1.copy()
    B = R2.copy()
    _nb_update(R, B, True)
    _nb_signfix(R)
    return R


@_jit
def _nb_solve_R(R, beta):
    p = R.shape[0] - 1
    for i in range(p):
        if R[i, i] == 0.0:
            beta[:] = np.nan
            return np.nan
    for i in range(p - 1, -1, -1):
        s = R[i, p]
        for j in range(i + 1, p):
            s -= R[i, j] * beta[j]
        beta[i] = s / R[i, i]
    return R[p, p] * R[p, p]


@_jit
def _nb_solve_G(G, beta):
    p = G.shape[0] - 1
    Gxx = np.ascontiguousarray(G[:p, :p])
    gxy = np.ascontiguousarray(G[:p, p])
    try:
        x = np.linalg.solve(Gxx, gxy)
    except Exception:  # matriz singular: mismo comportamiento que solve_G (NaN)
        beta[:] = np.nan
        return np.nan
    beta[:] = x
    return G[p, p] - np.dot(gxy, x)


@_jit
def _nb_count_sliding(nblocks, m, t_start, b, emit_from):
    c = 0
    for k in range(nblocks):
        if k + 1 >= m and t_start + (k + 1) * b - 1 >= emit_from:
            c += 1
    return c


@_jit
def _nb_sliding_qr(A, t_start, b, W, emit_from):
    n = A.shape[1]
    p = n - 1
    m = W // b
    nblocks = A.shape[0] // b
    nout = _nb_count_sliding(nblocks, m, t_start, b, emit_from)
    ends = np.empty(nout, np.int64)
    betas = np.empty((nout, p))
    rss = np.empty(nout)
    back = np.empty((m + 1, n, n))
    front = np.empty((m + 1, n, n))
    back_agg = np.zeros((n, n))
    nbk = 0
    nf = 0
    o = 0
    for k in range(nblocks):
        Rb = _nb_leaf(A[k * b:(k + 1) * b])
        back[nbk] = Rb
        back_agg = _nb_combine(back_agg, Rb) if nbk > 0 else Rb.copy()
        nbk += 1
        if nf + nbk > m:
            if nf == 0:  # volteo: agregados sufijo del más nuevo al más viejo
                acc = back[nbk - 1].copy()
                front[0] = acc
                for i in range(nbk - 2, -1, -1):
                    acc = _nb_combine(back[i], acc)
                    front[nbk - 1 - i] = acc
                nf = nbk
                nbk = 0
            nf -= 1
        e = t_start + (k + 1) * b - 1
        if nf + nbk == m and e >= emit_from:
            if nf > 0 and nbk > 0:
                Q = _nb_combine(front[nf - 1], back_agg)
            elif nf > 0:
                Q = front[nf - 1]
            else:
                Q = back_agg
            rss[o] = _nb_solve_R(Q, betas[o])
            ends[o] = e
            o += 1
    return ends, betas, rss


@_jit
def _nb_tree_range(T, M, lo, hi, acc, has):
    lo += M
    hi += M + 1
    while lo < hi:
        if lo & 1:
            if has:
                acc = _nb_combine(acc, T[lo])
            else:
                acc = T[lo].copy()
                has = True
            lo += 1
        if hi & 1:
            hi -= 1
            if has:
                acc = _nb_combine(acc, T[hi])
            else:
                acc = T[hi].copy()
                has = True
        lo //= 2
        hi //= 2
    return acc, has


@_jit
def _nb_sliding_qrtree(A, t_start, b, W, emit_from):
    n = A.shape[1]
    p = n - 1
    m = W // b
    M = 1
    while M < m:
        M *= 2
    nblocks = A.shape[0] // b
    nout = _nb_count_sliding(nblocks, m, t_start, b, emit_from)
    ends = np.empty(nout, np.int64)
    betas = np.empty((nout, p))
    rss = np.empty(nout)
    T = np.zeros((2 * M, n, n))  # R nulo = elemento neutro exacto de ⊕ por la derecha
    o = 0
    for k in range(nblocks):
        j = M + (k % M)
        T[j] = _nb_leaf(A[k * b:(k + 1) * b])
        j //= 2
        while j >= 1:
            T[j] = _nb_combine(T[2 * j], T[2 * j + 1])
            j //= 2
        e = t_start + (k + 1) * b - 1
        if k + 1 >= m and e >= emit_from:
            first = k + 1 - m
            if m == M:
                Q = T[1]
            else:
                lo = first % M
                hi = k % M
                acc = np.zeros((n, n))
                if lo <= hi:
                    acc, has = _nb_tree_range(T, M, lo, hi, acc, False)
                else:
                    acc, has = _nb_tree_range(T, M, lo, M - 1, acc, False)
                    acc, has = _nb_tree_range(T, M, 0, hi, acc, has)
                Q = acc
            rss[o] = _nb_solve_R(Q, betas[o])
            ends[o] = e
            o += 1
    return ends, betas, rss


@_jit
def _nb_sliding_ne(A, t_start, b, W, emit_from, reset, do_reset):
    n = A.shape[1]
    p = n - 1
    m = W // b
    nblocks = A.shape[0] // b
    nout = _nb_count_sliding(nblocks, m, t_start, b, emit_from)
    ends = np.empty(nout, np.int64)
    betas = np.empty((nout, p))
    rss = np.empty(nout)
    ring = np.empty((m + 1, n, n))
    G = np.zeros((n, n))
    o = 0
    for k in range(nblocks):
        blk = A[k * b:(k + 1) * b]
        Gb = np.dot(blk.T, blk)
        ring[k % (m + 1)] = Gb
        G += Gb
        if k + 1 > m:
            G -= ring[(k - m) % (m + 1)]
        full = k + 1 >= m
        if do_reset and full and reset > 0 and k % reset == 0:
            Wd = A[(k + 1 - m) * b:(k + 1) * b]
            G = np.dot(Wd.T, Wd)
        e = t_start + (k + 1) * b - 1
        if full and e >= emit_from:
            rss[o] = _nb_solve_G(G, betas[o])
            ends[o] = e
            o += 1
    return ends, betas, rss


@_jit
def _nb_expanding(A, t0, b, method, R0, has_R0, G0, has_G0, min_rows):
    """method: 0 = qr (contador binario), 1 = qr-seq (actualización secuencial), 2 = ne."""
    n = A.shape[1]
    p = n - 1
    t1 = t0 + A.shape[0] - 1
    e0 = ((t0 + b) // b) * b - 1
    nout = 0
    e = e0
    while e <= t1:
        if e + 1 >= min_rows:
            nout += 1
        e += b
    ends = np.empty(nout, np.int64)
    betas = np.empty((nout, p))
    rss = np.empty(nout)
    stack = np.empty((64, n, n))
    lv = np.empty(64, np.int64)
    top = 0
    R = R0.copy()
    hasR = has_R0
    G = G0.copy() if has_G0 else np.zeros((n, n))
    pos = 0
    o = 0
    e = e0
    while e <= t1:
        seg = A[pos:e - t0 + 1]
        pos = e - t0 + 1
        if method == 0:
            Rn = _nb_leaf(seg)
            lvl = 0
            while top > 0 and lv[top - 1] == lvl:
                Rn = _nb_combine(stack[top - 1], Rn)
                top -= 1
                lvl += 1
            stack[top] = Rn
            lv[top] = lvl
            top += 1
        elif method == 1:
            if hasR:
                B = seg.copy()
                _nb_update(R, B, False)
                _nb_signfix(R)
            else:
                R = _nb_leaf(seg)
                hasR = True
        else:
            G += np.dot(seg.T, seg)
        if e + 1 >= min_rows:
            if method == 0:
                acc = stack[top - 1].copy()
                for i in range(top - 2, -1, -1):
                    acc = _nb_combine(stack[i], acc)
                if has_R0:
                    acc = _nb_combine(R0, acc)
                rss[o] = _nb_solve_R(acc, betas[o])
            elif method == 1:
                rss[o] = _nb_solve_R(R, betas[o])
            else:
                rss[o] = _nb_solve_G(G, betas[o])
            ends[o] = e
            o += 1
        e += b
    return ends, betas, rss


def _to_tuples(ends, betas, rss):
    return [(int(ends[i]), betas[i], float(rss[i])) for i in range(len(ends))]


def numba_warmup(n=4):
    """Compila (o carga de la caché) todos los kernels con un problema mínimo bien planteado.

    Los tipos de Numba no dependen de n; las ventanas tienen más filas que columnas para
    que ningún sistema sea singular.
    """
    if not HAS_NUMBA:
        return
    W = 10 * max(n, 4)
    A = np.ascontiguousarray(np.random.default_rng(0).standard_normal((4 * W, n)))
    Z = np.zeros((n, n))
    _nb_sliding_qr(A, 0, 5, W, 0)
    _nb_sliding_qrtree(A, 0, 5, W, 0)
    _nb_sliding_ne(A, 0, 5, W, 0, 1, True)
    for mcode in (0, 1, 2):
        _nb_expanding(A, 0, 5, mcode, Z, False, Z, False, W)


def _resolve_engine(engine):
    if engine == "auto":
        return "numba" if HAS_NUMBA else "numpy"
    if engine == "numba" and not HAS_NUMBA:
        raise RuntimeError("numba no está instalado (pip install numba)")
    return engine


# ------------------------------------------------------------------- ventanas


def _check_bw(b, W=None):
    if not isinstance(b, (int, np.integer)) or b <= 0:
        raise ValueError("b debe ser un entero positivo")
    if W is not None and (W <= 0 or W % b):
        raise ValueError("W debe ser positivo y múltiplo de b")


def aligned_ends(t0, t1, b):
    e0 = ((t0 + b) // b) * b - 1
    return range(e0, t1 + 1, b)


def sliding_windows(A, t_start, b, W, method="qr", reset=0, emit_from=0, engine="numpy"):
    """Ventanas [e-W+1, e] con (e+1) % b == 0. Requiere t_start % b == 0 y W % b == 0.

    engine: "numpy" (referencia legible), "numba" (kernels compilados para qr, ne, ne-reset;
    givens y naive siempre usan numpy) o "auto".
    """
    _check_bw(b, W)
    if t_start % b:
        raise ValueError("t_start debe estar alineado a un bloque global")
    if method not in SLIDING_METHODS:
        raise ValueError(f"método {method} no soportado en ventana deslizante")
    if _resolve_engine(engine) == "numba" and method in NUMBA_SLIDING:
        A = np.ascontiguousarray(A, dtype=np.float64)
        if method == "qr":
            return _to_tuples(*_nb_sliding_qr(A, int(t_start), int(b), int(W), int(emit_from)))
        if method == "qr-tree":
            return _to_tuples(*_nb_sliding_qrtree(A, int(t_start), int(b), int(W), int(emit_from)))
        return _to_tuples(*_nb_sliding_ne(A, int(t_start), int(b), int(W), int(emit_from),
                                          int(reset), method == "ne-reset"))
    m = W // b
    nb = len(A) // b
    out = []
    ts, G, q, Rg, nfail = TwoStacksR(), None, deque(), None, 0
    st = RingSegTreeR(m) if method == "qr-tree" else None
    for k in range(nb):
        blk = A[k * b:(k + 1) * b]
        e = t_start + (k + 1) * b - 1
        if method == "qr":
            ts.push(r_factor(blk))
            if len(ts) > m:
                ts.evict()
            full = len(ts) == m
        elif method == "qr-tree":
            st.set(k, r_factor(blk))
            full = k + 1 >= m
        elif method in ("ne", "ne-reset"):
            Gb = blk.T @ blk
            q.append(Gb)
            G = Gb.copy() if G is None else G + Gb
            if len(q) > m:
                G = G - q.popleft()
            full = len(q) == m
            if method == "ne-reset" and full and reset > 0 and k % reset == 0:
                Wd = A[(k + 1 - m) * b:(k + 1) * b]
                G = Wd.T @ Wd
        elif method == "givens":
            full = k + 1 >= m
            try:
                if Rg is None:
                    Rg = r_factor(blk)
                else:
                    for row in blk:
                        Rg = _givens_update(Rg, row)
                if k + 1 > m:
                    for row in A[(k - m) * b:(k + 1 - m) * b]:
                        Rg = _downdate(Rg, row)
            except np.linalg.LinAlgError:
                nfail += 1
                Rg = r_factor(A[max(0, k + 1 - m) * b:(k + 1) * b])
        else:  # naive
            full = k + 1 >= m
        if full and e >= emit_from:
            if method == "qr":
                beta, rss = solve_R(ts.query())
            elif method == "qr-tree":
                beta, rss = solve_R(st.query(k + 1 - m, k))
            elif method in ("ne", "ne-reset"):
                beta, rss = solve_G(G)
            elif method == "givens":
                beta, rss = solve_R(Rg)
            else:
                beta, rss = solve_lstsq(A[(k + 1 - m) * b:(k + 1) * b])
            out.append((e, beta, rss))
    if method == "givens" and nfail:
        print(f"  [givens] downdates fallidos con recálculo: {nfail}", file=sys.stderr)
    return out


def expanding_windows(A, t0, b, method="qr", R0=None, G0=None, min_rows=1, engine="numpy"):
    """Ventanas [0, e] para (e+1) % b == 0, partiendo del prefijo R0/G0 de las filas < t0.

    qr:     contador binario sobre los segmentos locales; prefijo = R0 ⊕ fold(contador)
    qr-seq: R <- R([R; segmento]), actualización QR secuencial clásica (una cadena de k pasos)
    ne:     G <- G + segᵀseg
    """
    _check_bw(b)
    if method not in EXPANDING_METHODS:
        raise ValueError(f"método {method} no soportado en ventana expansiva")
    if _resolve_engine(engine) == "numba":
        A = np.ascontiguousarray(A, dtype=np.float64)
        n = A.shape[1]
        Z = np.zeros((n, n))
        code = {"qr": 0, "qr-seq": 1, "ne": 2}[method]
        return _to_tuples(*_nb_expanding(
            A, int(t0), int(b), code,
            Z if R0 is None else np.ascontiguousarray(R0), R0 is not None,
            Z if G0 is None else np.ascontiguousarray(G0), G0 is not None, int(min_rows)))
    t1 = t0 + len(A) - 1
    out, pos = [], 0
    bc, R, G = BinaryCounterR(), R0, G0
    for e in aligned_ends(t0, t1, b):
        seg = A[pos:e - t0 + 1]
        pos = e - t0 + 1
        if method == "qr":
            bc.push(r_factor(seg))
        elif method == "qr-seq":
            R = r_factor(seg) if R is None else r_factor(np.vstack([R, seg]))
        else:
            Gs = seg.T @ seg
            G = Gs if G is None else G + Gs
        if e + 1 >= min_rows:
            if method == "qr":
                beta, rss = solve_R(combine(R0, bc.fold()))
            elif method == "qr-seq":
                beta, rss = solve_R(R)
            else:
                beta, rss = solve_G(G)
            out.append((e, beta, rss))
    return out


def design(X, y, intercept=True):
    cols = [np.ones((len(y), 1))] if intercept else []
    return np.column_stack(cols + [X, y.reshape(-1, 1)]).astype(np.float64)


# ------------------------------------------- lógica distribuida (sin dependencia de Spark)


def partition_summary(t, A, W, b, mode, method):
    """Pase 1 de una partición: extremos, cola de W+b filas y, en expanding, su R o G."""
    if len(t) == 0:
        return None
    if not np.all(np.diff(t) == 1):
        raise RuntimeError("filas no contiguas en la partición")
    k = min(len(t), W + b)
    s = {"t0": int(t[0]), "t1": int(t[-1]),
         "tail": np.column_stack([t[-k:].astype(np.float64), A[-k:]]),
         "R": None, "G": None}
    if mode == "expanding":
        if method in ("qr", "qr-seq"):
            s["R"] = r_factor(A)
        else:
            s["G"] = A.T @ A
    return s


def plan_partitions(summaries, W, b, mode, method):
    """Driver: halos (sliding) y prefijos exclusivos (expanding) por partición.

    summaries: lista de dicts de partition_summary con clave 'pid', en cualquier orden.
    Devuelve ({pid: (R0, G0, s, H)}, bytes_a_difundir).
    Prefijo entre particiones: contador binario para qr, cadena para qr-seq, suma para ne.
    """
    _check_bw(b, W)
    summ = sorted(summaries, key=lambda d: d["t0"])
    if not summ:
        raise RuntimeError("no hay datos")
    if summ[0]["t0"] != 0:
        raise RuntimeError("t debe comenzar en 0")
    ncol = summ[0]["tail"].shape[1]
    info, buf = {}, np.empty((0, ncol))
    bc, preR, preG, prev_t1 = BinaryCounterR(), None, None, -1
    for d in summ:
        t0 = d["t0"]
        if t0 != prev_t1 + 1:
            raise RuntimeError(f"hueco o traslape en t entre particiones: {prev_t1} -> {t0}")
        prev_t1 = d["t1"]
        e_min = ((t0 + b) // b) * b - 1
        s = max(0, e_min - W + 1)
        halo = buf[buf[:, 0] >= s] if len(buf) else buf
        if mode == "sliding" and s < t0:
            if len(halo) == 0 or int(halo[0, 0]) != s or int(halo[-1, 0]) != t0 - 1:
                raise RuntimeError(f"halo incompleto para partición {d['pid']}")
        R0 = None
        if mode == "expanding":
            R0 = bc.fold() if method == "qr" else preR
        info[d["pid"]] = (R0, preG, s, halo[:, 1:].copy() if mode == "sliding" else np.empty((0, ncol - 1)))
        if mode == "expanding":
            if method == "qr":
                bc.push(d["R"])
            elif method == "qr-seq":
                preR = combine(preR, d["R"])
            else:
                preG = d["G"] if preG is None else preG + d["G"]
        buf = np.vstack([buf, d["tail"]])[-(W + b):]
    nbytes = sum((0 if v[0] is None else v[0].nbytes) + (0 if v[1] is None else v[1].nbytes) + v[3].nbytes
                 for v in info.values())
    return info, int(nbytes)


def partition_compute(t, A, entry, W, b, mode, method, reset=0, engine="numpy"):
    """Pase 2 de una partición: ventanas cuyo extremo e cae en [t0, t1]."""
    if len(t) == 0:
        return []
    R0, G0, s, H = entry
    t0 = int(t[0])
    if mode == "sliding":
        if s < t0:
            if len(H) == 0:
                raise RuntimeError("halo vacío")
            local = A
        else:  # s == t0 (W >= b garantiza s <= t0)
            local = A[t >= s]
        Af = np.vstack([H, local]) if len(H) else local
        return sliding_windows(Af, s, b, W, method, reset, emit_from=t0, engine=engine)
    return expanding_windows(A, t0, b, method, R0, G0, min_rows=W, engine=engine)


# ------------------------------------------------------------------------- gen


def arx_chunks(n, phi=(1.2, -0.25), lags=6, exog=4, rho=0.999, level=1e4, const=1.0,
               sigma=1.0, burn=2000, chunk=1_000_000, seed=12345):
    """Genera (t, y, X) por bloques de una ARX con exógenas AR(1) de nivel `level` (controla κ).

    Devuelve (gamma, iterador). X = [y_{t-1..t-lags}, z_1..z_exog].
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    phi = np.asarray(phi, dtype=float)
    if lags < len(phi):
        raise ValueError("lags debe ser >= len(phi)")
    burn = max(burn, lags)
    gamma = rng.standard_normal(exog) * 0.5

    def it():
        a_y = np.r_[1.0, -phi]
        zi_y = np.zeros(len(phi))
        zi_z = np.zeros((exog, 1))
        yprev = np.zeros(0)
        total_raw, raw_done, t_next = n + burn, 0, 0
        while raw_done < total_raw:
            c = min(chunk, total_raw - raw_done)
            z = np.empty((exog, c))
            for k in range(exog):
                z[k], zi_z[k] = lfilter([1.0], [1.0, -rho], rng.standard_normal(c), zi=zi_z[k])
            z += level
            u = const + gamma @ z + sigma * rng.standard_normal(c)
            y, zi_y_new = lfilter([1.0], a_y, u, zi=zi_y)
            zi_y[:] = zi_y_new
            yy = np.concatenate([yprev, y])
            off = len(yprev)
            ii = np.arange(c)[(raw_done + np.arange(c)) >= burn]
            if len(ii):
                lag = np.column_stack([yy[off + ii - j] for j in range(1, lags + 1)])
                X = np.column_stack([lag, z[:, ii].T])
                yield t_next + np.arange(len(ii), dtype=np.int64), y[ii], X
                t_next += len(ii)
            raw_done += c
            yprev = yy[-lags:]

    return gamma, it()


def cmd_gen(a):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if a.n <= 0:
        sys.exit("--n debe ser positivo")
    gamma, chunks = arx_chunks(a.n, a.phi, a.lags, a.exog, a.rho, a.level, a.const,
                               a.sigma, a.burn, a.chunk, a.seed)
    os.makedirs(a.out, exist_ok=True)
    names = ["t", "y"] + [f"x{j}" for j in range(a.lags + a.exog)]
    schema = pa.schema([("t", pa.int64())] + [(c, pa.float64()) for c in names[1:]])
    writer = pq.ParquetWriter(os.path.join(a.out, "part-00000.parquet"), schema)
    rows = 0
    for t, y, X in chunks:
        cols = [pa.array(t), pa.array(y)] + [pa.array(X[:, j]) for j in range(X.shape[1])]
        writer.write_table(pa.Table.from_arrays(cols, schema=schema))
        rows += len(t)
    writer.close()
    meta = dict(vars(a), gamma=gamma.tolist(), rows=int(rows))
    meta.pop("func", None)
    with open(os.path.join(a.out, "_gen_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"filas: {rows}  columnas X: {a.lags + a.exog}  -> {a.out}")


def load_numpy(path, intercept, max_rows=None):
    import pyarrow.parquet as pq

    tb = pq.read_table(path)
    xcols = sorted([c for c in tb.column_names if c.startswith("x")], key=lambda s: int(s[1:]))
    t = tb.column("t").to_numpy()
    order = np.argsort(t)
    if max_rows:
        order = order[:max_rows]
    X = np.column_stack([tb.column(c).to_numpy() for c in xcols])[order]
    y = tb.column("y").to_numpy()[order]
    t = t[order]
    if not (t[0] == 0 and np.all(np.diff(t) == 1)):
        raise RuntimeError("t debe ser contiguo desde 0")
    return t, design(X, y, intercept)


def rel_err(B, Bref):
    B, Bref = np.atleast_2d(B), np.atleast_2d(Bref)
    return np.linalg.norm(B - Bref, axis=1) / np.linalg.norm(Bref, axis=1)


def _default_methods(mode, given, allowed):
    if given:
        ms = [m.strip() for m in given.split(",") if m.strip()]
    else:
        ms = ["qr", "qr-tree", "ne", "ne-reset", "givens"] if mode == "sliding" else ["qr", "qr-seq", "ne"]
    bad = [m for m in ms if m not in allowed]
    if bad:
        sys.exit(f"métodos no válidos para {mode}: {bad}")
    return ms


# ----------------------------------------------------------------------- check


def cmd_check(a):
    _check_bw(a.block, a.window)
    t, A = load_numpy(a.data, a.intercept, a.max_rows)
    n_rows, p = len(A), A.shape[1] - 1
    W, b = a.window, a.block
    print(f"filas={n_rows}  p={p}  W={W}  b={b}  cond(X ventana inicial)={np.linalg.cond(A[:W, :-1]):.3e}")
    allowed = SLIDING_METHODS if a.mode == "sliding" else EXPANDING_METHODS
    methods = [m for m in _default_methods(a.mode, a.methods, allowed) if m != "naive"]
    engine = _resolve_engine(a.engine)
    if engine == "numba":
        numba_warmup(A.shape[1])
    print(f"motor: {engine}")
    res, times = {}, {}
    if a.mode == "sliding":
        for mth in ["naive"] + methods:
            t0 = time.perf_counter()
            res[mth] = sliding_windows(A, 0, b, W, mth, a.reset, engine=engine)
            times[mth] = time.perf_counter() - t0
            print(f"  {mth:9s} ventanas={len(res[mth]):7d}  tiempo={times[mth]:9.3f} s")
    else:
        ends = [e for e in aligned_ends(0, n_rows - 1, b) if e + 1 >= W]
        sample = set(ends[:: max(1, len(ends) // a.exp_ref_points)]) | {ends[-1]}
        t0 = time.perf_counter()
        res["naive"] = [(e,) + solve_lstsq(A[:e + 1]) for e in sorted(sample)]
        times["naive"] = time.perf_counter() - t0
        print(f"  naive     ventanas={len(res['naive']):7d}  tiempo={times['naive']:9.3f} s (muestra)")
        for mth in methods:
            t0 = time.perf_counter()
            out = expanding_windows(A, 0, b, mth, min_rows=W, engine=engine)
            times[mth] = time.perf_counter() - t0
            res[mth] = [r for r in out if r[0] in sample]
            print(f"  {mth:9s} ventanas={len(out):7d}  tiempo={times[mth]:9.3f} s")

    ref_e = np.array([r[0] for r in res["naive"]])
    Bref = np.vstack([r[1] for r in res["naive"]])
    table = {"t_end": ref_e}
    print("\nerror relativo ||b - b_lstsq|| / ||b_lstsq||")
    print(f"  {'método':9s} {'mediana':>11s} {'máximo':>11s} {'1er 10%':>11s} {'último 10%':>11s}")
    for mth, out in res.items():
        if mth == "naive":
            continue
        e = np.array([r[0] for r in out])
        assert np.array_equal(e, ref_e), mth
        err = rel_err(np.vstack([r[1] for r in out]), Bref)
        table[mth] = err
        k = max(1, len(err) // 10)
        print(f"  {mth:9s} {np.nanmedian(err):11.3e} {np.nanmax(err):11.3e} "
              f"{np.nanmedian(err[:k]):11.3e} {np.nanmedian(err[-k:]):11.3e}")
    if a.csv:
        import pandas as pd

        pd.DataFrame(table).to_csv(a.csv, index=False)
        pd.DataFrame([times]).to_csv(a.csv.replace(".csv", "_tiempos.csv"), index=False)
        print(f"\nerrores por ventana -> {a.csv}")


# ---------------------------------------------------------------------- verify


def _mp_gram_rows(A, lo, hi, G=None):
    """Suma en precisión mpmath de las filas A[lo:hi] en la triangular superior de AᵀA."""
    import mpmath as mp

    n = A.shape[1]
    iu = [(i, j) for i in range(n) for j in range(i, n)]
    if G is None:
        G = [mp.mpf(0)] * len(iu)
    for r in A[lo:hi]:
        v = [mp.mpf(float(x)) for x in r]
        for k, (i, j) in enumerate(iu):
            G[k] += v[i] * v[j]
    return G


def _mp_solve(G, n):
    import mpmath as mp

    p = n - 1
    M = mp.matrix(n, n)
    k = 0
    for i in range(n):
        for j in range(i, n):
            M[i, j] = M[j, i] = G[k]
            k += 1
    bx = mp.lu_solve(M[:p, :p], M[:p, p])
    return np.array([float(v) for v in bx])


def reference_mp(A, ends, W, mode, dps=60):
    """Coeficientes de referencia en `dps` dígitos: ecuaciones normales con aritmética exacta efectiva.

    Los productos de dos float64 caben en 106 bits; con dps=60 (~199 bits) las sumas son
    prácticamente exactas, y κ(XᵀX) ~ 1e18 queda muy por debajo de 10^dps.
    """
    import mpmath as mp

    mp.mp.dps = dps
    n = A.shape[1]
    ref, G, row = {}, None, 0
    for e in sorted(ends):
        if mode == "sliding":
            ref[e] = _mp_solve(_mp_gram_rows(A, e - W + 1, e + 1), n)
        else:
            G = _mp_gram_rows(A, row, e + 1, G)
            row = e + 1
            ref[e] = _mp_solve(G, n)
    return ref


def cmd_verify(a):
    _check_bw(a.block, a.window)
    t, A = load_numpy(a.data, a.intercept, a.max_rows)
    W, b = a.window, a.block
    allowed = SLIDING_METHODS if a.mode == "sliding" else EXPANDING_METHODS
    methods = [m for m in _default_methods(a.mode, a.methods, allowed) if m != "naive"]
    ends = [e for e in aligned_ends(0, len(A) - 1, b) if e + 1 >= W]
    pts = sorted(set(ends[:: max(1, len(ends) // a.points)]) | {ends[-1]})

    engine = _resolve_engine(a.engine)
    print(f"motor: {engine}")
    est = {}
    for mth in methods:
        out = (sliding_windows(A, 0, b, W, mth, a.reset, engine=engine) if a.mode == "sliding"
               else expanding_windows(A, 0, b, mth, min_rows=W, engine=engine))
        d = {e: beta for e, beta, _ in out}
        est[mth] = {e: d[e] for e in pts}
    est["lstsq"] = {e: solve_lstsq(A[max(0, e - W + 1) if a.mode == "sliding" else 0:e + 1])[0] for e in pts}
    if a.spark_out:
        import pandas as pd

        d = pd.read_parquet(a.spark_out)
        d = dict(zip(d.t_end.to_numpy(), d.beta.to_numpy()))
        missing = [e for e in pts if e not in d]
        if missing:
            sys.exit(f"--spark-out no contiene las ventanas {missing[:5]}...")
        est["spark"] = {e: np.asarray(d[e]) for e in pts}

    t0 = time.perf_counter()
    ref = reference_mp(A, pts, W, a.mode, a.dps)
    print(f"referencia mpmath ({a.dps} dígitos) en {len(pts)} ventanas: {time.perf_counter() - t0:.1f} s\n")

    cols = ["lstsq"] + methods + (["spark"] if a.spark_out else [])
    print(f"{'t_end':>8s} {'cond(X)':>10s} " + " ".join(f"{c:>10s}" for c in cols))
    errs = {c: [] for c in cols}
    for e in pts:
        lo = e - W + 1 if a.mode == "sliding" else 0
        kappa = np.linalg.cond(A[lo:e + 1, :-1])
        row = []
        for c in cols:
            v = rel_err(est[c][e], ref[e])[0]
            errs[c].append(v)
            row.append(f"{v:10.2e}")
        print(f"{e:8d} {kappa:10.2e} " + " ".join(row))
    print(f"{'máximo':>8s} {'':>10s} " + " ".join(f"{np.max(errs[c]):10.2e}" for c in cols))
    if a.csv:
        import pandas as pd

        pd.DataFrame({"t_end": pts, **errs}).to_csv(a.csv, index=False)
        print(f"\nerrores -> {a.csv}")


# ----------------------------------------------------------------------- bench


def sliding_arrays(A, b, W, method, reset=0, engine="numba"):
    """Como sliding_windows sobre toda la serie, pero devuelve arreglos (ends, betas, rss)."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    if _resolve_engine(engine) == "numba" and method in NUMBA_SLIDING:
        if method == "qr":
            return _nb_sliding_qr(A, 0, b, W, 0)
        if method == "qr-tree":
            return _nb_sliding_qrtree(A, 0, b, W, 0)
        return _nb_sliding_ne(A, 0, b, W, 0, reset, method == "ne-reset")
    out = sliding_windows(A, 0, b, W, method, reset, engine=engine)
    return (np.array([o[0] for o in out], dtype=np.int64), np.vstack([o[1] for o in out]),
            np.array([o[2] for o in out]))


def expanding_arrays(A, b, W, method, engine="numba"):
    A = np.ascontiguousarray(A, dtype=np.float64)
    if _resolve_engine(engine) == "numba":
        n = A.shape[1]
        Z = np.zeros((n, n))
        return _nb_expanding(A, 0, b, {"qr": 0, "qr-seq": 1, "ne": 2}[method], Z, False, Z, False, W)
    out = expanding_windows(A, 0, b, method, min_rows=W, engine=engine)
    return (np.array([o[0] for o in out], dtype=np.int64), np.vstack([o[1] for o in out]),
            np.array([o[2] for o in out]))


BENCH_FIELDS = ["mode", "method", "engine", "N", "p", "W", "b", "m", "level", "seed",
                "kappa_first", "kappa_median", "nwin", "t_min", "t_median", "repeats",
                "t_per_window_us", "extrapolated", "err_points", "err_median", "err_max",
                "err_first10", "err_last10", "mp_points", "err_mp_median", "err_mp_max"]


def _bench_key(d):
    return (d["mode"], d["method"], str(d["engine"]), int(d["N"]), int(d["W"]), int(d["b"]),
            float(d["level"]), int(d["seed"]))


def _err_summary(B, Bref):
    err = rel_err(B, Bref)
    k = max(1, len(err) // 10)
    return dict(err_points=len(err), err_median=float(np.nanmedian(err)), err_max=float(np.nanmax(err)),
                err_first10=float(np.nanmedian(err[:k])), err_last10=float(np.nanmedian(err[-k:])))


def cmd_bench(a):
    import csv

    engine = _resolve_engine(a.engine)
    if engine == "numba":
        numba_warmup()
    else:
        print("aviso: motor numpy; con N grande es muy lento", file=sys.stderr)
    modes = ["sliding", "expanding"] if a.mode == "both" else [a.mode]
    smeth = _default_methods("sliding", a.sliding_methods, SLIDING_METHODS)
    emeth = _default_methods("expanding", a.expanding_methods, EXPANDING_METHODS)
    done = set()
    if a.resume and os.path.exists(a.csv):
        with open(a.csv) as f:
            done = {_bench_key(r) for r in csv.DictReader(f)}
        print(f"reanudando: {len(done)} filas ya en {a.csv}")
    new_file = not os.path.exists(a.csv) or not a.resume
    fh = open(a.csv, "w" if new_file else "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=BENCH_FIELDS)
    if new_file:
        wr.writeheader()
        fh.flush()

    def emit(row):
        wr.writerow(row)
        fh.flush()
        mp = f"  mp_max={row['err_mp_max']:.2e}" if row["mp_points"] else ""
        print(f"  {row['mode']:9s} {row['method']:8s} N={row['N']:>9d} W={row['W']:>7d} "
              f"level={row['level']:.0e} seed={row['seed']}  t={row['t_median']:9.3f} s"
              f"{' (extrap.)' if row['extrapolated'] else ''}  err_med={row['err_median']:.2e} "
              f"err_max={row['err_max']:.2e}{mp}", flush=True)

    def timed(fn, reps):
        ts, out = [], None
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            ts.append(time.perf_counter() - t0)
        return out, float(np.min(ts)), float(np.median(ts))

    for N in a.n:
        for level in a.levels:
            for seed in a.seeds:
                t0 = time.perf_counter()
                _, chunks = arx_chunks(N, level=level, seed=seed)
                t_, y_, X_ = (np.concatenate(v) for v in zip(*chunks))
                A = np.ascontiguousarray(design(X_, y_, a.intercept))
                del t_, y_, X_
                p = A.shape[1] - 1
                print(f"datos N={N} level={level:.0e} seed={seed} generados en "
                      f"{time.perf_counter() - t0:.1f} s", flush=True)
                for b in a.blocks:
                    cfgs = []
                    if "sliding" in modes:
                        cfgs += [("sliding", W) for W in a.windows if W % b == 0 and W <= N // 2]
                    if "expanding" in modes:  # una sola corrida por serie: no depende de W
                        cfgs.append(("expanding", a.exp_min_rows))
                    for mode, W in cfgs:
                        base = dict(engine=engine, N=N, p=p, W=W, b=b, m=W // b if mode == "sliding" else 0,
                                    level=level, seed=seed, repeats=a.repeats)
                        if True:
                            ends = [e for e in aligned_ends(0, N - 1, b) if e + 1 >= W]
                            nwin = len(ends)
                            idx = np.unique(np.linspace(0, nwin - 1, min(a.err_points, nwin)).astype(int))
                            pts = np.array(ends)[idx]
                            lo = (lambda e: e - W + 1) if mode == "sliding" else (lambda e: 0)
                            kap = [np.linalg.cond(A[lo(e):e + 1, :-1]) for e in pts[:: max(1, len(pts) // 10)]]
                            base.update(kappa_first=float(np.linalg.cond(A[:W, :-1])),
                                        kappa_median=float(np.median(kap)), nwin=nwin)
                            Bref = np.vstack([solve_lstsq(A[lo(e):e + 1])[0] for e in pts])
                            mp_pts, mp_ref = np.array([], dtype=np.int64), None
                            if a.mp_points > 0:
                                sel = np.unique(np.linspace(0, len(pts) - 1, min(a.mp_points, len(pts))).astype(int))
                                mp_pts = pts[sel]
                                tm = time.perf_counter()
                                mp_ref = reference_mp(A, list(mp_pts), W, mode, a.dps)
                                print(f"  referencia mpmath ({len(mp_pts)} ventanas, {mode}, W={W}): "
                                      f"{time.perf_counter() - tm:.1f} s", flush=True)

                            def mp_summary(get_beta):
                                if mp_ref is None:
                                    return dict(mp_points=0, err_mp_median=np.nan, err_mp_max=np.nan)
                                err = np.array([rel_err(get_beta(e), mp_ref[e])[0] for e in mp_pts])
                                return dict(mp_points=len(err), err_mp_median=float(np.median(err)),
                                            err_mp_max=float(np.max(err)))
                            methods = smeth if mode == "sliding" else emeth
                            for mth in (["naive"] if mode == "sliding" else []) + methods:
                                key = (mode, mth, engine, N, W, b, float(level), seed)
                                if key in done:
                                    continue
                                row = dict(base, mode=mode, method=mth, extrapolated=0)
                                if mth == "naive":
                                    k = min(nwin, a.naive_sample)
                                    sel = np.array(ends)[np.unique(np.linspace(0, nwin - 1, k).astype(int))]
                                    _, tmin, tmed = timed(lambda: [solve_lstsq(A[e - W + 1:e + 1]) for e in sel],
                                                          a.repeats)
                                    f = nwin / len(sel)
                                    row.update(t_min=tmin * f, t_median=tmed * f, extrapolated=int(len(sel) < nwin),
                                               err_points=len(pts), err_median=0.0, err_max=0.0,
                                               err_first10=0.0, err_last10=0.0,
                                               **mp_summary(lambda e: Bref[np.searchsorted(pts, e)]))
                                else:
                                    fn = ((lambda: sliding_arrays(A, b, W, mth, a.reset, engine)) if mode == "sliding"
                                          else (lambda: expanding_arrays(A, b, W, mth, engine)))
                                    (e_out, B, _), tmin, tmed = timed(fn, a.repeats)
                                    pos = np.searchsorted(e_out, pts)
                                    assert np.array_equal(e_out[pos], pts)
                                    row.update(t_min=tmin, t_median=tmed, **_err_summary(B[pos], Bref),
                                               **mp_summary(lambda e: B[np.searchsorted(e_out, e)]))
                                    del e_out, B
                                row["t_per_window_us"] = 1e6 * row["t_median"] / nwin
                                emit(row)
                del A
    fh.close()
    print(f"resultados -> {a.csv}")


# ------------------------------------------------------------------------- run


def cmd_run(a):
    import pandas as pd
    from pyspark import TaskContext
    from pyspark.sql import SparkSession

    b, W, method, mode, intercept, reset = a.block, a.window, a.method, a.mode, a.intercept, a.reset
    if b <= 0:
        sys.exit("--block debe ser un entero positivo")
    if W <= 0 or W % b:
        sys.exit("--window debe ser positivo y múltiplo de --block")
    if a.partitions <= 0:
        sys.exit("--partitions debe ser un entero positivo")
    allowed = DIST_SLIDING_METHODS if mode == "sliding" else DIST_EXPANDING_METHODS
    if method not in allowed:
        sys.exit(f"--method {method} no soportado en modo {mode} distribuido; opciones: {allowed}")

    engine = _resolve_engine(a.engine)
    here = os.path.dirname(os.path.abspath(__file__))
    if not a.ship_module:
        # Los workers importan este mismo archivo (misma ruta) y reutilizan la caché Numba
        # de su __pycache__, que el driver llena aquí antes de lanzar las tareas.
        os.environ["PYTHONPATH"] = here + os.pathsep + os.environ.get("PYTHONPATH", "")
    if engine == "numba":
        numba_warmup()
    builder = SparkSession.builder.appName(f"swqr-{mode}-{method}-{engine}")
    if a.master:
        builder = builder.master(a.master)
    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "100000")
    if a.ship_module:
        # Clúster sin ruta compartida: se envía el archivo; cada worker compila los kernels
        # una vez (la caché no se reutiliza entre aplicaciones porque la ruta cambia).
        spark.sparkContext.addPyFile(os.path.abspath(__file__))

    T0 = time.perf_counter()
    df = spark.read.parquet(a.data)
    xcols = sorted([c for c in df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
    df = (df.select("t", *xcols, "y")
          .repartitionByRange(a.partitions, "t")
          .sortWithinPartitions("t")
          .cache())
    n_rows = df.count()
    T1 = time.perf_counter()
    n = len(xcols) + 1 + (1 if intercept else 0)

    def to_arrays(it, S):
        frames = [f for f in it if len(f)]
        if not frames:
            return None, None
        pdf = pd.concat(frames, ignore_index=True)
        t = pdf["t"].to_numpy(np.int64)
        return t, S.design(pdf[xcols].to_numpy(np.float64), pdf["y"].to_numpy(np.float64), intercept)

    def pass1(it):
        import swqr_spark as S

        t, A = to_arrays(it, S)
        if t is None:
            return
        s = S.partition_summary(t, A, W, b, mode, method)
        yield pd.DataFrame({
            "pid": [TaskContext.get().partitionId()], "t0": [s["t0"]], "t1": [s["t1"]],
            "tail": [s["tail"].ravel().tolist()],
            "R": [[] if s["R"] is None else s["R"].ravel().tolist()],
            "G": [[] if s["G"] is None else s["G"].ravel().tolist()]})

    rows1 = df.mapInPandas(pass1, "pid int, t0 long, t1 long, tail array<double>, "
                                  "R array<double>, G array<double>").collect()
    T2 = time.perf_counter()

    summaries = [{"pid": r.pid, "t0": r.t0, "t1": r.t1,
                  "tail": np.array(r.tail).reshape(-1, n + 1),
                  "R": np.array(r.R).reshape(n, n) if len(r.R) else None,
                  "G": np.array(r.G).reshape(n, n) if len(r.G) else None} for r in rows1]
    info, comm_bytes = plan_partitions(summaries, W, b, mode, method)
    bc = spark.sparkContext.broadcast(info)
    T3 = time.perf_counter()

    def pass2(it):
        import swqr_spark as S

        t, A = to_arrays(it, S)
        if t is None:
            return
        out = S.partition_compute(t, A, bc.value[TaskContext.get().partitionId()], W, b, mode, method,
                                  reset, engine)
        if out:
            yield pd.DataFrame({"t_end": [o[0] for o in out],
                                "beta": [o[1].tolist() for o in out],
                                "rss": [o[2] for o in out]})

    df.mapInPandas(pass2, "t_end long, beta array<double>, rss double").write.mode("overwrite").parquet(a.out)
    T4 = time.perf_counter()
    n_win = spark.read.parquet(a.out).count()

    summary = dict(mode=mode, method=method, engine=engine, rows=n_rows, p=n - 1, W=W, b=b,
                   partitions=df.rdd.getNumPartitions(), windows=n_win,
                   t_load_cache=T1 - T0, t_pass1=T2 - T1, t_driver=T3 - T2,
                   t_pass2_write=T4 - T3, t_total=T4 - T0,
                   bytes_broadcast=comm_bytes,
                   master=spark.sparkContext.master,
                   default_parallelism=spark.sparkContext.defaultParallelism)
    print(json.dumps(summary, indent=2))
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(summary) + "\n")
    spark.stop()


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=1_000_000)
    g.add_argument("--phi", type=float, nargs="+", default=[1.2, -0.25])
    g.add_argument("--lags", type=int, default=6)
    g.add_argument("--exog", type=int, default=4)
    g.add_argument("--rho", type=float, default=0.999, help="AR(1) de exógenas")
    g.add_argument("--level", type=float, default=1e4, help="nivel de exógenas (controla κ)")
    g.add_argument("--const", type=float, default=1.0)
    g.add_argument("--sigma", type=float, default=1.0)
    g.add_argument("--burn", type=int, default=2000)
    g.add_argument("--chunk", type=int, default=1_000_000)
    g.add_argument("--seed", type=int, default=12345)

    for name in ("check", "verify", "run"):
        c = sub.add_parser(name)
        c.add_argument("--data", required=True)
        c.add_argument("--mode", choices=["sliding", "expanding"], default="sliding")
        c.add_argument("--window", type=int, default=1000, help="sliding: tamaño; expanding: mínimo de filas")
        c.add_argument("--block", type=int, default=10, help="paso entre ventanas evaluadas")
        c.add_argument("--reset", type=int, default=100, help="periodo de recálculo para ne-reset (ventanas)")
        c.add_argument("--no-intercept", dest="intercept", action="store_false")
        c.add_argument("--engine", choices=["auto", "numba", "numpy"], default="auto",
                       help="auto = numba si está instalado")
        if name in ("check", "verify"):
            c.add_argument("--methods", default=None,
                           help="lista separada por comas; por omisión según --mode")
            c.add_argument("--max-rows", type=int, default=None)
            c.add_argument("--csv", default=None)
        if name == "check":
            c.add_argument("--exp-ref-points", type=int, default=200)
        if name == "verify":
            c.add_argument("--points", type=int, default=12, help="ventanas de muestra")
            c.add_argument("--dps", type=int, default=60, help="dígitos decimales de mpmath")
            c.add_argument("--spark-out", default=None, help="salida Parquet de `run` a evaluar")
        if name == "run":
            c.add_argument("--method", default="qr")
            c.add_argument("--partitions", type=int, default=8)
            c.add_argument("--out", required=True)
            c.add_argument("--master", default=None, help="si no se usa spark-submit, p.ej. local[*]")
            c.add_argument("--log", default=None, help="archivo JSONL para acumular corridas")
            c.add_argument("--ship-module", action="store_true",
                           help="enviar este archivo a los workers (clúster sin ruta compartida)")

    bn = sub.add_parser("bench", help="barrido de tiempo y exactitud -> CSV")
    bn.add_argument("--csv", required=True)
    bn.add_argument("--mode", choices=["sliding", "expanding", "both"], default="both")
    bn.add_argument("--n", type=int, nargs="+", default=[1_000_000])
    bn.add_argument("--windows", type=int, nargs="+", default=[1000, 5000, 20000, 100000],
                    help="sliding: tamaño de ventana; expanding: filas mínimas")
    bn.add_argument("--blocks", type=int, nargs="+", default=[10])
    bn.add_argument("--levels", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5],
                    help="nivel de las exógenas (controla κ)")
    bn.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    bn.add_argument("--sliding-methods", default="qr,qr-tree,ne,ne-reset")
    bn.add_argument("--expanding-methods", default="qr,qr-seq,ne")
    bn.add_argument("--repeats", type=int, default=3)
    bn.add_argument("--err-points", type=int, default=50, help="ventanas de muestra para el error")
    bn.add_argument("--naive-sample", type=int, default=300,
                    help="ventanas en las que se cronometra lstsq; si hay más, se extrapola")
    bn.add_argument("--reset", type=int, default=100)
    bn.add_argument("--engine", choices=["auto", "numba", "numpy"], default="auto")
    bn.add_argument("--no-intercept", dest="intercept", action="store_false")
    bn.add_argument("--exp-min-rows", type=int, default=1000, help="expanding: filas mínimas del prefijo")
    bn.add_argument("--mp-points", type=int, default=0,
                    help="ventanas con referencia mpmath (lenta: ~W·p²/2 operaciones por ventana en "
                         "sliding y ~N·p²/2 en total en expanding); 0 = no calcular")
    bn.add_argument("--dps", type=int, default=60)
    bn.add_argument("--resume", action="store_true", help="omitir configuraciones ya presentes en --csv")

    a = ap.parse_args()
    {"gen": cmd_gen, "check": cmd_check, "verify": cmd_verify, "run": cmd_run, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
