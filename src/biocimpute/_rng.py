"""R-compatible random number generation.

Reproduces exactly what ``set.seed(seed); sample(seq(n), size)`` does in R
(>= 3.6.0) with the default ``RNGkind()``: Mersenne-Twister, Inversion,
Rejection. Only the pieces used by ``impute`` are implemented.

References (R sources): src/main/RNG.c (``Randomize``, ``FixupSeeds``,
``MT_genrand``, ``fixup``, ``rbits``, ``R_unif_index``) and
src/main/random.c (``do_sample``).
"""

import math

_N = 624
_M = 397
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_MATRIX_A = 0x9908B0DF
_I2_32M1 = 2.328306437080797e-10  # 1/(2^32 - 1)


class RRandom:
    """A private R Mersenne-Twister stream, seeded like ``set.seed(seed)``."""

    def __init__(self, seed):
        # RNG_Init: scramble the seed, then fill the 625-word seed vector.
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        i_seed = []
        for _ in range(_N + 1):
            s = (69069 * s + 1) & 0xFFFFFFFF
            i_seed.append(s)
        # FixupSeeds: dummy[0] = mti = N, so the first draw regenerates mt[].
        self._mti = _N
        self._mt = i_seed[1:]

    def _genrand(self):
        mt = self._mt
        if self._mti >= _N:
            for kk in range(_N):
                y = (mt[kk] & _UPPER_MASK) | (mt[(kk + 1) % _N] & _LOWER_MASK)
                mt[kk] = mt[(kk + _M) % _N] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y * 2.3283064365386963e-10  # reals: [0,1)-interval

    def unif_rand(self):
        """``unif_rand()``: a uniform on (0, 1), as R's ``fixup`` ensures."""
        value = self._genrand()
        if value <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - value <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return value

    def _rbits(self, bits):
        v = 0
        for _ in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
        # mask out the bits beyond `bits` in the result
        return v & ((1 << bits) - 1)

    def unif_index(self, dn):
        """``R_unif_index(dn)``: uniform integer on [0, dn) by rejection."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample(self, n, size):
        """``sample.int(n, size)`` without replacement, 1-based like R."""
        if size > n:
            raise ValueError(
                "cannot take a sample larger than the population when 'replace = FALSE'"
            )
        # R's swap-remove over x = 0..n-1; only swapped slots are stored.
        x = {}
        y = []
        for _ in range(size):
            j = self.unif_index(n)
            y.append(x.get(j, j) + 1)
            n -= 1
            x[j] = x.get(n, n)
        return y
