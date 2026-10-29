import numpy as np
import pytest

from biocimpute import _kernels

pytestmark = pytest.mark.usefixtures("backend")


def _porder(dist, kn):
    pos = np.full(kn, -1, dtype=np.int64)
    nndist = np.empty(kn)
    _kernels._backend.porder(kn, np.asarray(dist, dtype=float), len(dist), pos, nndist)
    return pos, nndist


def test_porder_is_stable_partial_sort():
    rng = np.random.default_rng(0)
    dist = rng.integers(0, 5, size=50).astype(float)
    for kn in (1, 3, 11, 50):
        pos, nndist = _porder(dist, kn)
        expected = np.argsort(dist, kind="stable")[:kn]
        np.testing.assert_array_equal(pos, expected)
        np.testing.assert_array_equal(nndist, dist[expected])


def test_porder_fills_only_available_slots():
    pos, _ = _porder([3.0, 1.0, 2.0], 5)
    np.testing.assert_array_equal(pos[:3], [1, 2, 0])
    np.testing.assert_array_equal(pos[3:], [-1, -1])


def test_misdis_averages_over_shared_columns(backend):
    x = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 5.0], [0.0, 0.0, 0.0]], order="F")
    imiss = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int32, order="F")
    dis = np.empty(3)
    cnt = np.empty(3, dtype=np.int32)
    backend.misdis(x[0].copy(), x, imiss[0].copy(), imiss, dis, cnt)
    np.testing.assert_array_equal(cnt, [2, 1, 0])
    np.testing.assert_array_equal(dis, [0.0, 4.0, backend.DISMAX])


def test_misave_flags_columns_without_observations():
    x = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 7.0]])
    imiss = np.array([[0, 1], [0, 1], [0, 0]])
    x0, imiss0 = _kernels.misave(x, imiss, [0, 1])
    np.testing.assert_array_equal(x0, [2.0, 0.0])
    np.testing.assert_array_equal(imiss0, [1, 2])


def test_knnimp_uses_nearest_rows():
    x = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.1, 1.2, 0.0], [10.0, 10.0, 10.0]]
    )
    imiss = np.zeros((4, 3), dtype=np.int32)
    imiss[2, 2] = 1
    irmiss = imiss.sum(axis=1)
    ximp, flags = _kernels.knnimp(x, imiss, irmiss, 1)
    assert ximp[2, 2] == 1.0
    np.testing.assert_array_equal(ximp[[0, 1, 3]], x[[0, 1, 3]])
    assert flags[2, 2] == 1


def test_twomis_separates_two_groups():
    rng = np.random.default_rng(1)
    x = np.vstack([rng.normal(-5, 1, (20, 4)), rng.normal(5, 1, (30, 4))])
    imiss = np.zeros(x.shape, dtype=np.int32)
    clust, nsize, ratio, it = _kernels.twomis(x, imiss, 5, 0.001, [0, 49])
    assert sorted(nsize) == [20, 30]
    assert set(clust[: nsize[0], 0]) == set(range(20))
    assert 1 <= it <= 5


def test_backend_selection():
    assert _kernels._load("rust").__name__ == "biocimpute._kernels._rust"
    with pytest.raises(ValueError, match="unknown BIOCIMPUTE_BACKEND"):
        _kernels._load("fortran")
