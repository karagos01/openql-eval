#!/usr/bin/env python3
"""Can octonion algebra be transformed into block matrices?  No, and not for a
reason that better engineering could fix.

The title and abstract of openQL/openOL both say the architecture works "by
transforming quaternion and octonion algebra into unified block matrices".  For
the quaternion half this is standard and the paper does it correctly
(see quaternion.py).  For the octonion half it is excluded by a theorem.

Matrix multiplication is associative.  Any algebra homomorphism phi from O into
matrices therefore satisfies phi([x,y,z]) = 0 for every associator
[x,y,z] = (xy)z - x(yz).  This script measures that the associators of the
author's own multiplication table span the entire 7-dimensional imaginary part of
O, so phi annihilates Im O and keeps only the real line: phi is never faithful,
at any matrix size, over any field, with any choice of basis or blocking.

The same point is then measured rather than argued.  The construction that works
for H is the left regular representation L_x, y -> xy, which is an 8x8 real
matrix for the octonions too -- it just is not multiplicative.  The script
measures how badly L_x L_y differs from L_{xy}, side by side with the quaternion
case where the difference is zero.

Reference [5] of the paper is Baez, "The Octonions", Bull. AMS 39 (2002), which
states the non-associativity of O on its first page.  The one source cited for
the octonion half of the architecture is the source that rules it out.
"""
import itertools

import numpy as np

from fano import PAPER, table

np.random.seed(0)
R = np.random.randn

SGN, IDX = table(PAPER)                      # SOTP Eq. (3), the author's own table
C = np.zeros((8, 8, 8))
for i in range(8):
    for j in range(8):
        C[i, j, IDX[i][j]] = SGN[i][j]

E = np.eye(8)
omul = lambda x, y: np.einsum("i,j,ijk->k", x, y, C)
L = lambda x: np.einsum("i,ijk->kj", x, C)   # left multiplication, 8x8 real matrix

# ---- 1. the associators span the whole imaginary part ---------------------
rows, nz = [], 0
for i, j, k in itertools.product(range(8), repeat=3):
    a = omul(omul(E[i], E[j]), E[k]) - omul(E[i], omul(E[j], E[k]))
    if np.abs(a).max() > 1e-12:
        nz += 1
        rows.append(a)
A = np.array(rows)
rank = np.linalg.matrix_rank(A)
print("Associators of the published multiplication table")
print(f"  basis triples (i,j,k) with [e_i,e_j,e_k] != 0   {nz} of {8**3}")
print(f"  rank of the span of those associators           {rank}")
print(f"  dim Im O                                        7")
print(f"  real part reached by any associator             {np.abs(A[:, 0]).max():.1e}")
print("  -> the associators span Im O exactly, and never touch the real axis.")

print("\nConsequence for any matrix form")
print("  Matrix multiplication is associative, so an algebra homomorphism")
print("  phi: O -> M_n(R) obeys phi((xy)z) = phi(x(yz)), i.e. phi([x,y,z]) = 0.")
print(f"  The associators span a {rank}-dimensional subspace, so ker phi contains")
print(f"  all of Im O and phi retains {8-rank} of 8 dimensions ({100*(8-rank)/8:.1f} %).")
print("  O is a simple algebra, so the only homomorphism into matrices is the")
print("  zero map.  'Octonion algebra as a block matrix' is not an engineering")
print("  problem; it has no solution at any n.")

# ---- 2. the same thing measured, against the quaternion case -------------
def Mq(q):
    a, b, c, d = q
    return np.array([[a, -b, -c, -d], [b, a, -d, c],
                     [c, d, a, -b], [d, -c, b, a]])


def qmul(q, r):
    a, b, c, d = q
    e, f, g, h = r
    return np.array([a*e - b*f - c*g - d*h, a*f + b*e + c*h - d*g,
                     a*g - b*h + c*e + d*f, a*h + b*g - c*f + d*e])


N = 5000
qerr = qrel = oerr = orel = 0.0
for _ in range(N):
    q, r = R(4), R(4)
    d = np.abs(Mq(q) @ Mq(r) - Mq(qmul(q, r))).max()
    qerr = max(qerr, d)
    qrel = max(qrel, d / np.abs(Mq(qmul(q, r))).max())
    x, y = R(8), R(8)
    d = np.abs(L(x) @ L(y) - L(omul(x, y))).max()
    oerr = max(oerr, d)
    orel = max(orel, d / np.abs(L(omul(x, y))).max())

print(f"\nLeft regular representation, {N} random pairs each")
print(f"  {'algebra':<10s} {'max |L_x L_y - L_xy|':>22s} {'relative':>12s}")
print(f"  {'H  (4x4)':<10s} {qerr:>22.2e} {qrel:>12.2e}")
print(f"  {'O  (8x8)':<10s} {oerr:>22.2e} {orel:>12.2e}")
print("  -> the identical construction is exact for H and wrong by order 1 for O.")

# where exactly it fails: alternativity makes the diagonal survive
diag = max(np.abs(L(E[i]) @ L(E[i]) - L(omul(E[i], E[i]))).max() for i in range(8))
offd = max(np.abs(L(E[i]) @ L(E[j]) - L(omul(E[i], E[j]))).max()
           for i in range(8) for j in range(8) if i != j)
print(f"\n  L_x L_x vs L_(xx) on basis units (alternativity)  {diag:.1e}")
print(f"  L_x L_y vs L_(xy) for x != y                      {offd:.1e}")
print("  -> O is alternative, so squares still work; every mixed product does not.")
print("     An MMA pipeline computes mixed products, which is the whole point of")
print("     a 'unified' node, so the failure is exactly where the paper needs it")
print("     to succeed.")
