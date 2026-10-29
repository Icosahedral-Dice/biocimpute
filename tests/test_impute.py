import logging

import numpy as np
import pandas as pd
import pytest

from biocimpute import ImputeKnnResult, impute_knn, load_khanmiss


pytestmark = pytest.mark.usefixtures("backend")

def test_khanmiss_layout():
    khan = load_khanmiss()
    assert khan.shape == (2309, 65)
    assert list(khan.columns[:3]) == ["X", "X1", "sample1"]
    assert khan.iloc[0, 2] == "EWS"
    assert khan.iloc[1:, 2:].isna().to_numpy().sum() == 1282


def test_result_and_defaults():
    x = np.array([[1.0, 2.0, 3.0], [1.5, np.nan, 3.5], [10.0, 20.0, 30.0]])
    result = impute_knn(x, k=1)
    assert isinstance(result, ImputeKnnResult)
    assert result.rng_seed == 362436069
    assert result.data[1, 1] == 2.0
    assert np.isnan(x[1, 1]), "input must not be modified"


def test_dataframe_round_trip():
    df = pd.DataFrame(
        [[1.0, 2.0], [1.5, None], [3.0, 4.0]],
        index=["g1", "g2", "g3"],
        columns=["s1", "s2"],
    )
    out = impute_knn(df, k=1).data
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["g1", "g2", "g3"]
    assert list(out.columns) == ["s1", "s2"]
    assert out.loc["g2", "s2"] == 2.0


def test_colmax_error():
    x = np.ones((10, 3))
    x[:9, 0] = np.nan
    with pytest.raises(ValueError, match="a column has more than 80 % missing values!"):
        impute_knn(x)


def test_rowmax_warning_uses_column_means():
    x = np.arange(12.0).reshape(4, 3)
    x[0, :2] = np.nan
    x[1, :2] = np.nan
    with pytest.warns(UserWarning, match="2 rows with more than 50 % entries missing"):
        out = impute_knn(x, k=1).data
    # means over the rows kept for knn (rows 2 and 3)
    np.testing.assert_array_equal(out[:2, :2], [[7.5, 8.5], [7.5, 8.5]])


def test_single_row_over_rowmax_keeps_zeros_like_r():
    # R's meanimp skips one-row blocks, leaving the zero fill in place.
    x = np.arange(12.0).reshape(4, 3) + 1
    x[0, :2] = np.nan
    with pytest.warns(UserWarning):
        out = impute_knn(x, k=1).data
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 3.0])


def test_fewer_rows_than_k_uses_all_other_rows():
    # R's Fortran reads out of bounds here; we average all other rows.
    x = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    out = impute_knn(x, k=10).data
    assert out[1, 1] == 4.0


def test_split_logs_cluster_sizes(caplog):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4))
    x[rng.random(x.shape) < 0.1] = np.nan
    caplog.set_level(logging.INFO, logger="biocimpute")
    impute_knn(x, k=3, maxp=40)
    assert caplog.records[0].getMessage().startswith("Cluster size 60 broken into")


def test_same_seed_same_result():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 5))
    x[rng.random(x.shape) < 0.15] = np.nan
    a = impute_knn(x, k=3, rowmax=0.8, maxp=20, rng_seed=1).data
    b = impute_knn(x, k=3, rowmax=0.8, maxp=20, rng_seed=1).data
    np.testing.assert_array_equal(a, b)


def test_rejects_non_matrix():
    with pytest.raises(ValueError, match="2-dimensional"):
        impute_knn([1.0, np.nan, 2.0])
