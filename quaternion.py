#!/usr/bin/env python3
"""Is the 4x4 real matrix in openQL Sec. 2.1 what the paper says it is, and what
does "concatenating three such state matrices" produce?

Sec. 2.1 prints a 4x4 matrix M(q) for a quaternion q = a + bi + cj + dk and calls
it isomorphic.  It is: this script verifies M(q)M(r) = M(qr) to machine precision.
That is the one algebraic step in the paper that is carried out correctly and
completely, and reference [1] (Kuipers) supports it.

The paper then builds a node by "concatenating three such state matrices" into a
single structure driven by one Matrix Multiply-Accumulate instruction, and calls
the result a "unified state structure".  Three 4x4 blocks in a 12x12 matrix is
the only reading that gives the 12x12 the paper needs, and that algebra is
H (+) H (+) H: the three quaternions evolve side by side and no entry of the
product couples block i to block j.  Nothing in the node is unified.
"""
import numpy as np

np.random.seed(0)
R = np.random.randn


def Mq(q):
    """openQL Sec. 2.1, verbatim."""
    a, b, c, d = q
    return np.array([[a, -b, -c, -d],
                     [b,  a, -d,  c],
                     [c,  d,  a, -b],
                     [d, -c,  b,  a]])


def qmul(q, r):
    """Hamilton product, written out independently of Mq."""
    a, b, c, d = q
    e, f, g, h = r
    return np.array([a*e - b*f - c*g - d*h,
                     a*f + b*e + c*h - d*g,
                     a*g - b*h + c*e + d*f,
                     a*h + b*g - c*f + d*e])


def block(qs):
    """Three 4x4 state matrices concatenated into one 12x12 node."""
    Z = np.zeros((4, 4))
    return np.block([[Mq(qs[0]), Z, Z],
                     [Z, Mq(qs[1]), Z],
                     [Z, Z, Mq(qs[2])]])


# ---- 1. the 4x4 matrix is a faithful representation of H -------------------
pairs = [(R(4), R(4)) for _ in range(5000)]
err = max(np.abs(Mq(q) @ Mq(r) - Mq(qmul(q, r))).max() for q, r in pairs)
scale = max(np.abs(Mq(qmul(q, r))).max() for q, r in pairs)
print("Sec. 2.1, the 4x4 matrix")
print(f"  M(q) M(r) = M(qr) over {len(pairs)} random pairs")
print(f"    max absolute error            {err:.2e}")
print(f"    relative to entry scale       {err/scale:.2e}")
print("  -> correct.  This is the left regular representation of H, and matrix")
print("     multiplication reproduces the Hamilton product exactly.  The claim")
print("     'isomorphic 4x4 real matrix' holds as written.")

# a representation is not free: it stores 4 numbers in 16 slots
print(f"\n  state carried by the node      {4} real numbers")
print(f"  slots the matrix occupies      {16}")
print(f"  redundancy                     {16/4:.0f} x")

# ---- 2. what the 12x12 node actually is -----------------------------------
print("\nSec. 2.1, 'concatenating three such state matrices'")
qs, rs = [R(4) for _ in range(3)], [R(4) for _ in range(3)]
prod = block(qs) @ block(rs)
want = block([qmul(qs[i], rs[i]) for i in range(3)])
print(f"  product equals blockwise (q_i r_i), max error  {np.abs(prod - want).max():.2e}")

# measure the coupling directly: perturb block 0 of the right operand only and
# see whether anything outside block 0 of the product moves
pert = [rs[0] + np.array([1.0, 0, 0, 0]), rs[1], rs[2]]
delta = np.abs(block(qs) @ block(pert) - prod)
off = delta.copy()
off[0:4, 0:4] = 0.0
print(f"  perturbing block 0 of the operand moves block 0 by  {delta[0:4,0:4].max():.3f}")
print(f"  and everything outside block 0 by                   {off.max():.3f}")

nz = sum(1 for i in range(12) for j in range(12) if abs(block(qs)[i, j]) > 1e-12)
print(f"\n  structurally zero entries      {144-48} of 144  ({100*(144-48)/144:.0f} %)")
print(f"  nonzero entries in one node    {nz} of 144")
print(f"  entries coupling block i to j  0")
print(f"  state carried by the node      12 real numbers")
print(f"  slots the matrix occupies      144")
print(f"  redundancy                     {144/12:.0f} x")
print("  -> the algebra is H (+) H (+) H.  The three quaternions cannot exchange")
print("     information, so the 12x12 node is three independent 4D systems in a")
print("     container 12 times larger than their state, not a unified field.")

# ---- 3. the two incompatible derivations of 12 -----------------------------
print("\nWhere does 12 come from?")
print("  Sec. 2.1   3 quaternions x 4 components                  = 12")
print("  Sec. 6     dim U(1) + dim SU(2) + dim SU(3) = 1 + 3 + 8  = 12")
print("  -> both appear in the paper and they are not the same 12: the first is")
print("     three copies of a 4D algebra, the second is the Lie algebra of a")
print("     12-dimensional group with no quaternionic block structure.  Nothing")
print("     in the paper connects them, and 112 is given no derivation at all")
print(f"     (112 = 28 x 4 = 14 x 8, neither of which the paper mentions).")
