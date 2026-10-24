"""Port of impute's R/allknnimp.R.

The functions mirror their R counterparts one to one (``impute.knn`` ->
``impute_knn``, ``knnimp.split`` -> ``_knnimp_split``, ...), including the
places where R's behavior is surprising, so that results match R:

* ``knnimp`` zero-fills missing entries before splitting, so the two-means
  clustering in ``twomeans.miss`` sees a fully observed, zero-filled matrix.
* Clusters with at most ``k`` rows go through ``meanimp`` on that zero-filled
  block, which is a no-op: their missing entries come back as 0.
* ``meanimp`` does nothing for a block with a single row, so a lone row with
  more than ``rowmax`` missing also keeps 0 in its missing entries.
"""

import logging
import math
import sys
import warnings
from typing import Any, NamedTuple

import numpy as np

from . import _kernels
from ._rng import RRandom

logger = logging.getLogger("biocimpute")


class ImputeKnnResult(NamedTuple):
    """Return value of :func:`impute_knn`.

    R's ``impute.knn`` also returns ``rng.state``, the global ``.Random.seed``
    before reseeding. biocimpute uses a private generator and leaves every
    global random state untouched, so there is nothing to restore.
    """

    data: Any
    """The imputed matrix (a DataFrame if a DataFrame was passed)."""
    rng_seed: int
    """The seed used, to reproduce the imputation."""


def impute_knn(data, k=10, rowmax=0.5, colmax=0.8, maxp=1500, rng_seed=362436069):
    """Impute missing expression data using nearest neighbor averaging.

    Port of ``impute::impute.knn``; see that function's documentation for the
    algorithm.

    Parameters
    ----------
    data : array_like or pandas.DataFrame
        An expression matrix with genes in the rows, samples in the columns.
        Missing values are NaN.
    k : int
        Number of neighbors to be used in the imputation.
    rowmax : float
        The maximum fraction of missing data allowed in any row. Rows with
        more missing are imputed using the overall mean per sample.
    colmax : float
        The maximum fraction of missing data allowed in any column. If any
        column has more missing data, a ``ValueError`` is raised.
    maxp : int
        The largest block of genes imputed using the knn algorithm; larger
        blocks are divided by two-means clustering (recursively) prior to
        imputation. If ``maxp`` equals the number of rows, only knn
        imputation is done.
    rng_seed : int
        The seed for the random number generator, for reproducibility. The
        same seed gives the same result as R's ``impute.knn``.

    Returns
    -------
    ImputeKnnResult
        ``data`` is the imputed matrix, ``rng_seed`` the seed used.
    """
    rng = RRandom(rng_seed)

    x, wrap = _as_matrix(data)
    p = x.shape[0]
    col_nas = np.isnan(x).sum(axis=0)
    if np.any(col_nas > colmax * p):
        raise ValueError(
            f"a column has more than {_r_format(round(colmax * 100))} % missing values!"
        )

    x = _knnimp(x, k, maxmiss=rowmax, maxp=maxp, rng=rng)

    return ImputeKnnResult(data=wrap(x), rng_seed=rng_seed)


def _knnimp(x, k, maxmiss, maxp, rng):
    pn = x.shape
    p, n = pn
    imiss = np.isnan(x)
    x = np.where(imiss, 0.0, x)
    irmiss = imiss.sum(axis=1)
    imax = irmiss > math.trunc(maxmiss * n)
    simax = int(imax.sum())
    if simax > 0:
        warnings.warn(
            f"{simax} rows with more than {_r_format(round(maxmiss * 100, 1))} "
            "% entries missing;\n mean imputation used for these rows",
            stacklevel=3,
        )
        irmiss = irmiss[~imax]
        imiss_omit = imiss[imax]
        imiss = imiss[~imax]
        xomit = x[imax]
        x = x[~imax]
        p = p - simax
    if p <= maxp:
        ximp = _knnimp_internal(x, k, imiss, irmiss, p, n, maxp, rng)
    else:
        ximp = _knnimp_split(x, k, imiss, irmiss, p, n, maxp, rng)
    imiss_new = np.isnan(ximp)
    newmiss = bool(imiss_new.any())
    if simax > 0 or newmiss:
        xbar = _mean_miss(x, imiss=imiss)
        if newmiss:
            ximp = _meanimp(ximp, imiss_new, xbar)
        if simax > 0:
            xomit = _meanimp(xomit, imiss_omit, xbar)
            xout = np.zeros(pn)
            xout[~imax] = ximp
            xout[imax] = xomit
            ximp = xout
    return ximp


def _knnimp_internal(x, k, imiss, irmiss, p, n, maxp, rng):
    if p <= maxp:
        ximp, imiss_out = _kernels.knnimp(x, imiss, irmiss, k)
        ximp[imiss_out == 2] = np.nan
        return ximp
    return _knnimp_split(x, k, imiss, irmiss, p, n, maxp, rng)


def _knnimp_split(x, k, imiss, irmiss, p, n, maxp, rng):
    x = x.copy()
    junk = _twomeans_miss(x, rng)
    size = junk["size"]
    logger.info("Cluster size %d broken into %d %d", p, size[0], size[1])
    clus = junk["cluster"]
    for i in range(2):
        p = int(size[i])
        index = clus == i + 1
        if p <= k:
            x[index] = _meanimp(x[index])
        else:
            x[index] = _knnimp_internal(
                x[index], k, imiss[index], irmiss[index], p, n, maxp, rng
            )
        logger.info("Done cluster %d", size[i])
    return x


def _mean_miss(x, index=None, imiss=None):
    if index is None:
        index = np.arange(x.shape[0])
    if imiss is None:
        imiss = np.isnan(x)
    x = np.where(imiss, 0.0, x)
    x0, imiss0 = _kernels.misave(x, imiss, index)
    x0[imiss0 == 2] = np.nan
    return x0


def _meanimp(x, imiss=None, xbar=None):
    if imiss is None:
        imiss = np.isnan(x)
    nr = x.shape[0]
    if nr > 1:
        if xbar is None:
            xbar = _mean_miss(x, imiss=imiss)
        x = x.copy()
        x[imiss] = np.broadcast_to(xbar, x.shape)[imiss]
    return x


def _twomeans_miss(x, rng, imiss=None, imbalance=0.2, maxit=5, eps=0.001):
    """Two-means cluster solution for data with missing entries. Returns a
    dict like R's list: cluster (1/2 per row), ratio, iter, size."""
    p = x.shape[0]
    if imiss is None:
        imiss = np.isnan(x)
        x = np.where(imiss, 0.0, x)
    starters = np.asarray(rng.sample(p, 2)) - 1
    clust, nsize, ratio, it = _kernels.twomis(x, imiss, maxit, eps, starters)
    cluster = np.ones(p, dtype=np.int64)
    cluster[clust[: nsize[1], 1]] = 2
    return {"cluster": cluster, "ratio": ratio, "iter": it, "size": nsize}


def _as_matrix(data):
    """Coerce input to a float64 matrix, plus a function that restores the
    DataFrame index and columns (R's dimnames) on the result."""
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(data, pd.DataFrame):
        x = data.to_numpy(dtype=np.float64, na_value=np.nan)
        index, columns = data.index, data.columns
        return x, lambda m: pd.DataFrame(m, index=index, columns=columns)
    x = np.array(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("data must be a 2-dimensional matrix")
    return x, lambda m: m


def _r_format(value):
    """Format a number the way R's ``format()`` does for these messages."""
    return f"{value:.7g}"
