"""Compare impute_knn with R's impute.knn on the fixtures written by
tests/r/generate_fixtures.R."""

import csv
import logging
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from biocimpute import impute_knn, load_khanmiss

FIXTURES = Path(__file__).parent / "fixtures"


def _read_manifest():
    with open(FIXTURES / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def _float(s):
    return np.nan if s == "NA" else float(s)


@cache
def _load_input(name):
    if name == "khan":
        return load_khanmiss().iloc[1:, 2:].to_numpy(dtype=float)
    with open(FIXTURES / f"{name}_input.csv", newline="") as f:
        return np.array([[_float(v) for v in row] for row in csv.reader(f)])


def _load_imputed(name):
    with open(FIXTURES / f"{name}_imputed.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    idx = np.array([[int(r["row"]) - 1, int(r["col"]) - 1] for r in rows])
    values = np.array([_float(r["value"]) for r in rows])
    return idx, values


SCENARIOS = _read_manifest()


@pytest.mark.parametrize("sc", SCENARIOS, ids=[sc["name"] for sc in SCENARIOS])
def test_matches_r(sc, caplog):
    x = _load_input(sc["input"])
    kwargs = dict(
        k=int(sc["k"]),
        rowmax=float(sc["rowmax"]),
        colmax=float(sc["colmax"]),
        maxp=int(sc["maxp"]),
        rng_seed=int(sc["rng_seed"]),
    )

    if sc["error"]:
        with pytest.raises(ValueError) as excinfo:
            impute_knn(x, **kwargs)
        assert str(excinfo.value) == sc["error"]
        return

    caplog.set_level(logging.INFO, logger="biocimpute")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = impute_knn(x, **kwargs)
    messages = [str(w.message) for w in caught]
    assert messages == ([sc["warning"]] if sc["warning"] else [])

    # R's cat() output, one line per message (cat adds a trailing space).
    r_lines = [line.rstrip() for line in sc["stdout"].splitlines()]
    assert [r.getMessage() for r in caplog.records] == r_lines

    out = result.data
    observed = ~np.isnan(x)
    np.testing.assert_array_equal(out[observed], x[observed])

    idx, expected = _load_imputed(sc["name"])
    np.testing.assert_array_equal(idx, np.argwhere(~observed.T)[:, ::-1])
    got = out[idx[:, 0], idx[:, 1]]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12, equal_nan=True)
