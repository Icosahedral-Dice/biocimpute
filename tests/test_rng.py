import pytest

from biocimpute._rng import RRandom

# Generated in R 4.5.2 with: set.seed(seed); sample.int(n, size)
R_SAMPLES = [
    (362436069, 2309, 2, [125, 1351]),
    (1, 10, 10, [9, 4, 7, 1, 2, 5, 3, 10, 6, 8]),
    (12345, 100, 7, [14, 51, 80, 90, 92, 24, 58]),
    (-5, 2308, 2, [1783, 233]),
    (0, 2, 2, [2, 1]),
    (42, 65536, 3, [61413, 54425, 34020]),
    (42, 65537, 3, [61413, 54425, 34020]),
    (7, 1000003, 4, [615891, 922076, 445122, 981210]),
    (2147483647, 2147483647, 3, [814677222, 845146519, 1908426088]),
    (-2147483647, 17, 5, [1, 12, 15, 9, 10]),
]


@pytest.mark.parametrize("seed, n, size, expected", R_SAMPLES)
def test_sample_matches_r(seed, n, size, expected):
    assert RRandom(seed).sample(n, size) == expected


def test_unif_rand_matches_r():
    # set.seed(99); sprintf("%.17g", runif(5))
    expected = [
        0.58471185155212879,
        0.1137816752307117,
        0.68426474370062351,
        0.99250877718441188,
        0.53499358450062573,
    ]
    rng = RRandom(99)
    assert [rng.unif_rand() for _ in range(5)] == expected


def test_stream_continues_across_calls():
    # set.seed(362436069); sample(seq(2309), 2); sample(seq(1000), 2); sample(seq(7), 2)
    rng = RRandom(362436069)
    assert rng.sample(2309, 2) == [125, 1351]
    assert rng.sample(1000, 2) == [720, 174]
    assert rng.sample(7, 2) == [4, 1]


def test_sample_larger_than_population():
    with pytest.raises(ValueError, match="larger than the population"):
        RRandom(1).sample(1, 2)
