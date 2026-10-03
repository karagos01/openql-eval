#!/usr/bin/env python3
"""Does the Fano-plane orientation printed in SOTP Eq. (3) define an octonion algebra?

The paper fixes the structure constants by listing seven oriented triads.  Only
some orientations of the seven Fano lines yield a normed division algebra; the
rest are merely 8-dimensional non-associative algebras with zero divisors.  This
script builds the multiplication table from the published triads and tests the
three properties the paper relies on: norm multiplicativity, alternativity, and
absence of zero divisors.
"""
import itertools
import numpy as np

# SOTP Eq. (3), verbatim, including the orientation of each triad.
PAPER = [(1, 2, 3), (1, 4, 5), (2, 4, 6), (3, 4, 7), (1, 7, 6), (2, 5, 7), (3, 6, 5)]

# The same seven lines with every triad written in ascending order instead.  This
# is the obvious-looking choice and it is NOT an octonion algebra, which is the
# point: the orientation carries the content, not the line set.
ASCENDING = [(1, 2, 3), (1, 4, 5), (1, 6, 7), (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 5, 6)]


def table(triads):
    """8x8 multiplication table from oriented triads; entry = (sign, index)."""
    t = np.zeros((8, 8), dtype=int)      # index of basis element
    s = np.zeros((8, 8), dtype=int)      # sign
    for i in range(8):
        t[0, i] = t[i, 0] = i
        s[0, i] = s[i, 0] = 1
    for m in range(1, 8):
        t[m, m] = 0
        s[m, m] = -1
    for a, b, c in triads:
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):   # cyclic: xy = +z
            t[x, y] = z; s[x, y] = 1
            t[y, x] = z; s[y, x] = -1
    return s, t


def mul(s, t, X, Y):
    Z = np.zeros(8)
    for i in range(8):
        if X[i] == 0.0:
            continue
        for j in range(8):
            if Y[j] == 0.0:
                continue
            Z[t[i, j]] += s[i, j] * X[i] * Y[j]
    return Z


def check(triads, n=4000, seed=0):
    s, t = table(triads)
    # every product must land on a basis element with a sign: table well formed?
    holes = int((t == 0).sum() - 8)          # e_m e_m = -1 gives the 7 legal zeros +e0e0
    rng = np.random.default_rng(seed)
    norm_err = assoc_zero = alt_err = 0
    worst_norm = 0.0
    for _ in range(n):
        X, Y, Z = rng.normal(size=(3, 8))
        XY = mul(s, t, X, Y)
        d = abs(np.linalg.norm(XY) - np.linalg.norm(X) * np.linalg.norm(Y))
        rel = d / (np.linalg.norm(X) * np.linalg.norm(Y))
        worst_norm = max(worst_norm, rel)
        if rel > 1e-9:
            norm_err += 1
        # alternativity: (XX)Y == X(XY)
        if np.linalg.norm(mul(s, t, mul(s, t, X, X), Y)
                          - mul(s, t, X, mul(s, t, X, Y))) > 1e-9 * np.linalg.norm(X) ** 2 * np.linalg.norm(Y):
            alt_err += 1
        # associator must not vanish identically (paper's premise)
        A = mul(s, t, mul(s, t, X, Y), Z) - mul(s, t, X, mul(s, t, Y, Z))
        if np.linalg.norm(A) < 1e-12:
            assoc_zero += 1
    return dict(holes=holes, norm_violations=norm_err, worst_norm_rel=worst_norm,
                alternativity_violations=alt_err, associator_vanished=assoc_zero, n=n)


def zero_divisor(triads, tries=200000, seed=1):
    """Look for X,Y != 0 with XY = 0 (impossible in a division algebra)."""
    s, t = table(triads)
    rng = np.random.default_rng(seed)
    # search in the 2-dim planes spanned by pairs of basis units, where zero
    # divisors of a badly oriented table show up analytically
    best = (np.inf, None, None)
    for i, j in itertools.combinations(range(1, 8), 2):
        for k, l in itertools.combinations(range(1, 8), 2):
            for a in (1.0, -1.0):
                for b in (1.0, -1.0):
                    X = np.zeros(8); X[i] = 1.0; X[j] = a
                    Y = np.zeros(8); Y[k] = 1.0; Y[l] = b
                    n = np.linalg.norm(mul(s, t, X, Y))
                    if n < best[0]:
                        best = (n, X.copy(), Y.copy())
    return best


def enumerate_orientations():
    """How many of the 2^7 orientations of these 7 lines give a division algebra?"""
    lines = [tuple(sorted(l)) for l in ASCENDING]
    ok = []
    for bits in range(128):
        triads = []
        for n, (a, b, c) in enumerate(lines):
            triads.append((a, b, c) if (bits >> n) & 1 else (a, c, b))
        r = check(triads, n=60, seed=7)
        if r["norm_violations"] == 0 and r["alternativity_violations"] == 0:
            ok.append(tuple(triads))
    return ok


if __name__ == "__main__":
    print("=== SOTP Eq. (3), as published ===")
    for k, v in check(PAPER).items():
        print(f"  {k:28s} {v}")
    print("=== same 7 lines, every triad in ascending order ===")
    for k, v in check(ASCENDING).items():
        print(f"  {k:28s} {v}")

    n, X, Y = zero_divisor(PAPER)
    print(f"\nsmallest |XY| over basis pairs, published table:   {n:.3e}")
    if n < 1e-12:
        nz = [f"{'+' if X[i]>0 else '-'}e{i}" for i in range(8) if X[i]]
        nw = [f"{'+' if Y[i]>0 else '-'}e{i}" for i in range(8) if Y[i]]
        print(f"  zero divisor found: ({' '.join(nz)}) * ({' '.join(nw)}) = 0")
        print("  |X| =", np.linalg.norm(X), " |Y| =", np.linalg.norm(Y))
    n2, _, _ = zero_divisor(ASCENDING)
    print(f"smallest |XY| over basis pairs, ascending order:   {n2:.3e}")

    ok = enumerate_orientations()
    print(f"\norientations of the 7 Fano lines giving a normed division algebra: "
          f"{len(ok)} of 128")
    print("published orientation is among them:",
          tuple(PAPER) in {tuple(sorted(t, key=lambda x: x)) for t in ok}
          or any(sorted(map(tuple, o)) == sorted(map(tuple, PAPER)) for o in ok))
