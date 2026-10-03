#!/usr/bin/env python3
"""What does a 112x112 node per voxel cost, and what do the hardware claims in
openQL Sec. 3-5 come to when the arithmetic is done?

The claims checked here, verbatim from the paper:

  Sec. 2.2  "A single node operates with a 112x112 block matrix."
  Sec. 2.3  "the tensor product (x) is executed in a single clock cycle"
  Sec. 3    "keeps Tensor Core load in the efficient 15-20% range when
             simulating environments with millions of nodes"
  Sec. 4    "the memory footprint of a single node reaches tens of kilobytes"
  Sec. 5    "Materials Engineering: Searching for superconductors ... in the
             full 112D space of openOL via molecular brute-force simulation"

Reference numbers are the RTX 3090 the author's other preprints specify, taken
from the catalogue: 936 GB/s, 142 TFLOP/s fp16 on the tensor cores, 1.70 GHz
boost.  No GPU is required to run this script; it is arithmetic on published
specifications.
"""
import numpy as np

BW = 936e9           # bytes/s, catalogue read bandwidth
TC = 142e12          # FLOP/s, fp16 tensor-core peak
CLK = 1.70e9         # Hz, boost clock
VRAM = 24 * 2**30    # bytes
FP16 = 2             # bytes per value
D = 112

node = D * D * FP16                 # the paper's node: a 112x112 matrix
state = D * FP16                    # the physics it encodes: 112 potentials

print("1. What a node stores")
print(f"   112x112 fp16 block matrix        {node/1024:>10.1f} KiB"
      f"   <- 'tens of kilobytes', consistent")
print(f"   the 112 physical potentials      {state/1024:>10.3f} KiB")
print(f"   overhead of the matrix form      {node/state:>10.0f} x")
print(f"   nonzero fraction if built from 28 quaternion 4x4 blocks"
      f"   {28*16/(D*D):>6.2%}")

# the roofline position is unchanged by the representation
mm_flop, mm_bytes = 2 * D**3, 2 * node          # W @ S, read S write S'
mv_flop, mv_bytes = 2 * D**2, 2 * state         # W @ s, read s write s'
print("\n2. The matrix form buys no arithmetic intensity")
print(f"   {'operation':<26s} {'FLOP':>12s} {'bytes':>9s} {'FLOP/byte':>10s}")
print(f"   {'W @ S  (112x112 node)':<26s} {mm_flop:>12,} {mm_bytes:>9,}"
      f" {mm_flop/mm_bytes:>10.1f}")
print(f"   {'W @ s  (112-vector)':<26s} {mv_flop:>12,} {mv_bytes:>9,}"
      f" {mv_flop/mv_bytes:>10.1f}")
print(f"   RTX 3090 ridge point {TC/BW:.0f} FLOP/byte -> both are memory bound")
print(f"   -> identical position on the roofline, {mm_flop/mv_flop:.0f} x the FLOPs")
print(f"      and {mm_bytes/mv_bytes:.0f} x the traffic.  The block matrix is overhead,")
print("      not a path to the tensor cores.")

print("\n3. 'executed in a single clock cycle'")
chip_flop_per_cycle = TC / CLK
print(f"   whole-chip fp16 throughput       {chip_flop_per_cycle:>12,.0f} FLOP/cycle")
print(f"   one node's W @ S                 {mm_flop:>12,} FLOP")
print(f"   cycles, entire GPU on one node   {mm_flop/chip_flop_per_cycle:>12.1f}")
kron = (D * D) ** 2
print(f"   and if (x) is read as written, a Kronecker product of two 112x112")
print(f"   matrices has {kron:,} entries = {kron*FP16/2**20:,.0f} MiB per node.")

print("\n4. Resident nodes and the step rate")
print(f"   {'grid':>9s} {'nodes':>13s} {'S,S+1':>11s} {'x 24 GB':>8s}"
      f" {'cards':>6s} {'s/step':>8s} {'Hz':>7s}")
for g in (32, 64, 80, 100, 128, 256):
    n = g**3
    tot = 2 * n * node
    t = tot / BW
    print(f"   {g:>6d}^3 {n:>13,} {tot/2**30:>8.1f} GiB {tot/VRAM:>7.1f}x"
          f" {np.ceil(tot/VRAM):>6.0f} {t:>8.3f} {1/t:>7.1f}")
cap = VRAM / (2 * node)
fps60 = BW / (2 * node * 60)
print(f"\n   nodes whose state alone fills 24 GB     {cap:>12,.0f}"
      f"  ({cap**(1/3):.0f}^3)")
print(f"   nodes the bandwidth allows at 60 Hz     {fps60:>12,.0f}"
      f"  ({fps60**(1/3):.0f}^3)")
print("   -> 'millions of nodes' needs 47 GiB per million for state alone, with")
print("      no weight matrices, no framebuffer and no rendering.  One card runs")
print(f"      a {fps60**(1/3):.0f}^3 grid at 60 Hz with zero pixels drawn.")

print("\n5. 'Tensor Core load in the efficient 15-20% range'")
for u in (0.15, 0.20):
    print(f"   at {u:.0%} load the tensor cores deliver {u*TC/1e12:>6.1f} TFLOP/s"
          f" and {(1-u)*TC/1e12:>6.1f} TFLOP/s sit idle")
print("   -> 15-20 % utilisation of the only unit the architecture depends on is")
print("      the symptom, not a design target.  The paper moves all physics onto")
print("      the tensor cores precisely so they are saturated; a pipeline that")
print("      leaves 80-85 % of them idle has not achieved that.  Item 2 above")
print("      gives the reason: at 56 FLOP/byte the node is memory bound, so the")
print("      tensor cores wait on VRAM no matter how the work is scheduled.")

print("\n6. 'molecular brute-force simulation' at one node per atom")
a_si = 0.5431e-9                      # silicon lattice constant, m
atoms_per_cell = 8
atoms = cap
vol = atoms / atoms_per_cell * a_si**3
print(f"   atoms resident in 24 GB          {atoms:>12,.0f}")
print(f"   as a cube of crystalline silicon {vol**(1/3)*1e9:>12.1f} nm on a side")
print(f"   atoms in one mole                {6.022e23:>12.3g}")
print(f"   cards for one mole of nodes      {6.022e23/atoms:>12.3g}")
print("   -> a 22 nm nanoparticle, state only, nothing else resident.  That is a")
print("      legitimate size for atomistic work, which is why the comparison")
print("      matters: molecular dynamics reaches it today with a few tens of")
print("      bytes per atom, not 24.5 KiB, and without claiming 112 dimensions.")
