"""Compiled workhorse routines, one wrapper per ``.Fortran()`` call in R.

This module is the only place that imports a backend. The wrappers prepare
column-major float64/int32 arrays and allocate outputs, so another backend
(e.g. Rust) only has to provide the same in-place kernels.
"""

import numpy as np

from . import _numba as _backend


def _as_x(x):
    return np.asfortranarray(x, dtype=np.float64)


def _as_flags(a):
    return np.asfortranarray(a, dtype=np.int32)


def knnimp(x, imiss, irmiss, k):
    """``.Fortran("knnimp", ...)``: returns ``(ximp, imiss)`` where
    ``imiss == 2`` marks entries no neighbor could fill."""
    x = _as_x(x)
    ximp = x.copy(order="F")
    imiss = _as_flags(imiss).copy(order="F")
    irmiss = np.ascontiguousarray(irmiss, dtype=np.int32)
    _backend.knnimp(x, ximp, imiss, irmiss, int(k))
    return ximp, imiss


def misave(x, imiss, index):
    """``.Fortran("misave", ...)`` with all of ``imiss0`` set: column means of
    rows ``index`` (0-based) over observed entries, and the ``imiss0`` flags
    (2 where a column has no observed entry)."""
    x = _as_x(x)
    n = x.shape[1]
    x0 = np.zeros(n)
    imiss0 = np.ones(n, dtype=np.int32)
    index = np.ascontiguousarray(index, dtype=np.int64)
    _backend.misave(x, x0, imiss0, _as_flags(imiss), index, len(index))
    return x0, imiss0


def twomis(x, imiss, maxit, eps, istart):
    """``.Fortran("twomis", ...)``: returns ``(clust, nsize, ratio, iter)``
    where ``clust[:nsize[i], i]`` are the 0-based rows of cluster ``i``."""
    x = _as_x(x)
    p = x.shape[0]
    clust = np.zeros((p, 2), dtype=np.int64, order="F")
    nsize = np.zeros(2, dtype=np.int64)
    istart = np.ascontiguousarray(istart, dtype=np.int64)
    ratio, it = _backend.twomis(
        x, _as_flags(imiss), int(maxit), float(eps), istart, clust, nsize
    )
    return clust, nsize, ratio, it
