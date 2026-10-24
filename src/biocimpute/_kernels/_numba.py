"""Numba port of impute's src/knnimpute.f.

Each function is a literal, 0-based translation of the Fortran subroutine of
the same name, keeping the loop and accumulation order so that floating point
results match. Missing-value flags follow the Fortran convention: 0 means
present, 1 means missing, and 2 means "could not be imputed".
"""

import numpy as np
from numba import njit

DISMAX = 1e10


@njit(cache=True)
def misdis(x0, x, imiss0, imiss, dis, iworkp):
    """Mean squared distance from ``x0`` to every row of ``x`` over the
    columns observed in both."""
    p, n = x.shape
    for j in range(p):
        dis[j] = 0.0
        iworkp[j] = 0
    for k in range(n):
        if imiss0[k] > 0:
            continue
        for j in range(p):
            if imiss[j, k] > 0:
                continue
            d = x0[k] - x[j, k]
            dis[j] = dis[j] + d * d
            iworkp[j] = iworkp[j] + 1
    for j in range(p):
        if iworkp[j] > 0:
            dis[j] = dis[j] / iworkp[j]
        else:
            dis[j] = DISMAX


@njit(cache=True)
def misave(x, x0, imiss0, imiss, index, m):
    """Column means of the rows ``index[:m]`` of ``x`` over observed entries,
    computed only where ``imiss0`` is set; sets ``imiss0[k] = 2`` when none
    of those rows is observed in column ``k``."""
    n = x.shape[1]
    for k in range(n):
        x0[k] = 0.0
        if imiss0[k] == 0:
            continue
        ktot = 0
        for j in range(m):
            jj = index[j]
            if imiss[jj, k] > 0:
                continue
            x0[k] = x0[k] + x[jj, k]
            ktot = ktot + 1
        if ktot > 0:
            x0[k] = x0[k] / ktot
        else:
            imiss0[k] = 2


@njit(cache=True)
def porder(kn, dist, ntr, pos, nndist):
    """Positions of the ``kn`` smallest of ``dist[:ntr]`` in increasing order,
    ties going to the lower index. Only ``min(kn, ntr)`` slots are filled."""
    for j in range(ntr):
        if j < kn:
            inserted = False
            for k in range(j):
                if dist[j] < nndist[k]:
                    for k1 in range(j - 1, k - 1, -1):
                        nndist[k1 + 1] = nndist[k1]
                        pos[k1 + 1] = pos[k1]
                    nndist[k] = dist[j]
                    pos[k] = j
                    inserted = True
                    break
            if not inserted:
                nndist[j] = dist[j]
                pos[j] = j
        else:
            if dist[j] >= nndist[kn - 1]:
                continue
            for k in range(kn):
                if dist[j] < nndist[k]:
                    for k1 in range(kn - 2, k - 1, -1):
                        nndist[k1 + 1] = nndist[k1]
                        pos[k1 + 1] = pos[k1]
                    nndist[k] = dist[j]
                    pos[k] = j
                    break


@njit(cache=True)
def knnimp(x, ximp, imiss, irmiss, kn):
    """KNN-impute every row with missing values. Writes into ``ximp`` and
    sets ``imiss[i, k] = 2`` where no neighbor was observed."""
    p, n = x.shape
    m = kn + 1
    workp = np.empty(p)
    iworkp = np.empty(p, dtype=np.int32)
    workn = np.empty(n)
    iworkn = np.empty(n, dtype=np.int32)
    pos = np.zeros(m, dtype=np.int64)
    nndist = np.empty(m)
    # Neighbors are positions 2..kn+1 of the ordering. Fortran reads past the
    # filled slots when p <= kn; we use only the p - 1 that exist.
    nnb = min(m, p) - 1
    for i in range(p):
        if irmiss[i] == 0:
            continue
        for j in range(n):
            workn[j] = x[i, j]
            iworkn[j] = imiss[i, j]
        misdis(workn, x, iworkn, imiss, workp, iworkp)
        porder(m, workp, p, pos, nndist)
        misave(x, workn, iworkn, imiss, pos[1:], nnb)
        for k in range(n):
            if iworkn[k] == 0:
                continue
            ximp[i, k] = workn[k]
            if iworkn[k] == 2:
                imiss[i, k] = 2


@njit(cache=True)
def twomis(x, imiss, maxit, eps, istart, clust, nsize):
    """Two-means clustering of the rows of ``x`` with missing entries,
    started from rows ``istart``. Fills ``clust[:nsize[i], i]`` with the
    members of cluster ``i``; returns ``(ratio, iter)``."""
    p, n = x.shape
    if maxit < 1:
        maxit = 5
    x0 = np.empty((n, 2))
    imiss0 = np.empty((n, 2), dtype=np.int32)
    dist = np.empty((p, 2))
    iworkp = np.empty(p, dtype=np.int32)
    iworkn = np.empty(n, dtype=np.int32)
    for i in range(2):
        for j in range(n):
            x0[j, i] = x[istart[i], j]
            imiss0[j, i] = imiss[istart[i], j]
    it = 0
    ratio = 10.0
    dold = 0.0
    while it < maxit and ratio > eps:
        it = it + 1
        for i in range(2):
            misdis(x0[:, i], x, imiss0[:, i], imiss, dist[:, i], iworkp)
            nsize[i] = 0
        dnew = 0.0
        for j in range(p):
            if dist[j, 0] < dist[j, 1]:
                imax = 0
            else:
                imax = 1
            clust[nsize[imax], imax] = j
            nsize[imax] = nsize[imax] + 1
            dnew = dnew + dist[j, imax]
        if dnew == 0.0:
            break
        if it == 1:
            dold = dnew * 10.0
        ratio = abs(dnew - dold) / dold
        dold = dnew
        for i in range(2):
            for j in range(n):
                iworkn[j] = 1
            misave(x, x0[:, i], iworkn, imiss, clust[:, i], nsize[i])
            for j in range(n):
                if iworkn[j] == 2:
                    imiss0[j, i] = 1
                else:
                    imiss0[j, i] = 0
    return ratio, it
