# Half of the title is a theorem away: an evaluation of openQL / openOL

**karagos01**, October 2026

An independent evaluation of M. Mazgal, *openQL / openOL: Tensor Matrix
Architecture for Unified Simulation of Physical Fields in 12D and 112D via GPU
Tensor Cores*, Zenodo, 3 October 2026, DOI
[10.5281/zenodo.23112560](https://doi.org/10.5281/zenodo.23112560) (CC BY 4.0).

## Abstract

The preprint proposes replacing polygon graphics APIs with a volumetric
architecture in which every voxel carries a 12×12 (openQL) or 112×112 (openOL)
real block matrix, obtained "by transforming quaternion and octonion algebra into
unified block matrices", and in which the time evolution runs on GPU tensor cores
as a symplectic integrator. This evaluation checks the parts that can be
computed.

The quaternion half is correct and is carried out completely: the 4×4 matrix
printed in §2.1 is the left regular representation of ℍ and satisfies
M(q)M(r) = M(qr) to 1.8·10⁻¹⁵. This is the first time a central algebraic step in
this author's preprints survives a check end to end, and the supporting reference
is cited accurately.

The octonion half is excluded by a theorem rather than by engineering difficulty.
Matrix multiplication is associative, so any algebra homomorphism φ: 𝕆 → Mₙ(ℝ)
must annihilate every associator; the associators of the author's own published
multiplication table span the full 7-dimensional imaginary part of 𝕆, so φ retains
1 of 8 dimensions and, 𝕆 being simple, is the zero map at every n. The identical
construction that is exact for ℍ is wrong by a relative 2.8 for 𝕆. The paper's only
citation covering octonions, Baez's survey, states the non-associativity on its
first page.

Three further defects are measured. The 12×12 node is block diagonal, so its
algebra is ℍ⊕ℍ⊕ℍ: perturbing one block moves nothing outside it, 96 of 144 entries
are structurally zero, and the "unified state structure" stores 12 real numbers in
144 slots. The update rule S(t+1) = σ(W·S(t) + b) is not symplectic — fitted by
least squares to exact one-step data from a pendulum it is accurate to 1.5·10⁻³
after one step and ends a long rollout at 10.2× the initial energy, where leapfrog,
with no weights and no training, holds 1.0000. And the block-matrix representation
sits at exactly the same point on the roofline as the 112-component state it
encodes, 56 FLOP/byte against a ridge of 152, while costing 112× the memory traffic
and 112× the FLOPs; a single 24 GB card holds 513 588 nodes of state and nothing
else, and the bandwidth permits a 68³ grid at 60 Hz with no pixels drawn.

Everything below is reproduced by `./run_all.sh` (pure CPU, no network, ~20 s).

## 1. What is correct

**The 4×4 quaternion matrix.** §2.1 prints, for q = a + bi + cj + dk,

```
M(q) =  [ a  -b  -c  -d ]
        [ b   a  -d   c ]
        [ c   d   a  -b ]
        [ d  -c   b   a ]
```

and calls it isomorphic. It is. `quaternion.py` computes the Hamilton product
independently and checks M(q)M(r) = M(qr) on 5000 random pairs: maximum absolute
error **1.78·10⁻¹⁵**, **1.51·10⁻¹⁶** relative to the entry scale. This is the
standard left regular representation of ℍ, the sign conventions in the printed
matrix are right, and reference [1] (Kuipers) is a correct source for it.

This matters beyond the single equation. Across the earlier preprints from this
author the algebra was either wrong or absent; in the September trilogy the
octonion structure constants were right but the physical mapping was not. Here one
complete algebraic statement is made and verified. It should be said plainly before
anything else.

**The node size is stated honestly.** §4 says "the memory footprint of a single
node reaches tens of kilobytes". A 112×112 fp16 matrix is 24.5 KiB. The paper does
not understate its own cost.

**The references are real.** Five are given, and all five exist and are
load-bearing works in their fields: Kuipers on quaternions, Raissi et al. on
PINNs, Markidis et al. on tensor-core programmability, Mildenhall et al. on NeRF,
Baez on the octonions. The earlier MPPT and CQFT/PCTP preprints had no
bibliography at all. §9 below is about what those references are made to carry,
which is a separate question from whether they exist.

## 2. The octonion half of the title cannot exist

The title and abstract both say the architecture proceeds "by transforming
quaternion **and octonion** algebra into unified block matrices", and the engine
named openOL is the octonion one. There is no octonion multiplication anywhere in
the body of the paper. There cannot be, and the reason is a theorem.

Matrix multiplication is associative. So for any algebra homomorphism
φ: 𝕆 → Mₙ(ℝ) and any x, y, z,

    φ([x,y,z]) = φ((xy)z) − φ(x(yz)) = (φ(x)φ(y))φ(z) − φ(x)(φ(y)φ(z)) = 0,

i.e. every associator lies in ker φ. `octonion.py` measures what that kernel
contains, using the author's own multiplication table — the oriented Fano triads
of SOTP Eq. (3), which the previous evaluation verified to be a valid normed
division algebra:

| | measured |
|---|---|
| basis triples (i,j,k) with [eᵢ,eⱼ,eₖ] ≠ 0 | **168 of 512** |
| rank of the span of those associators | **7** |
| dim Im 𝕆 | 7 |
| real component reached by any associator | 0.0 |

The associators span the imaginary part exactly. So ker φ ⊇ Im 𝕆, φ retains **1 of
8 dimensions (12.5 %)**, and since 𝕆 is a simple algebra the kernel must be 0 or
all of 𝕆 — it is all of 𝕆, and φ is the zero map. This holds at every matrix size,
over any field, under any choice of basis, and under any blocking scheme, so no
amount of engineering reaches it.

The same fact measured rather than argued: the construction that works for ℍ is the
left regular representation L_x: y ↦ xy, which for 𝕆 is a perfectly good 8×8 real
matrix — it is simply not multiplicative. Over 5000 random pairs each:

| algebra | max \|L_x L_y − L_{xy}\| | relative |
|---|---|---|
| ℍ (4×4) | 1.78·10⁻¹⁵ | 4.24·10⁻¹⁶ |
| 𝕆 (8×8) | **29.5** | **2.76** |

Where it fails is also where the architecture needs it. 𝕆 is alternative, so
L_x L_x = L_{xx} holds exactly (measured 0.0 on basis units), and every mixed
product fails (2.0 on basis units). An MMA pipeline over a "unified" node computes
mixed products; that is the entire purpose of putting several physical quantities
in one matrix.

Reference [5] is J. C. Baez, *The Octonions*, Bull. AMS **39** (2002) 145–205,
cited for "underlying Lie algebra and gauge group symmetries corresponding to the
12D/112D spatial mappings". That survey contains no 12-dimensional or
112-dimensional mapping, and it states the non-associativity of 𝕆 in its opening
section. The one source cited for the octonion half of the architecture is the
source that rules the octonion half out.

## 3. The 12D node is three quaternions that cannot interact

§2.1 forms a node by "concatenating three such state matrices", yielding "a
unified state structure (voxel) of a circuit or matter … enabling computation via a
single Matrix Multiply-Accumulate (MMA) instruction". Three 4×4 blocks is the only
reading that produces the 12×12 the 12D layer requires. `quaternion.py` measures
what that object is:

| | measured |
|---|---|
| product equals blockwise (qᵢrᵢ) | error 4.4·10⁻¹⁶ |
| perturbing block 0 of the operand moves block 0 by | 1.940 |
| …and moves everything outside block 0 by | **0.000** |
| structurally zero entries | **96 of 144 (67 %)** |
| entries coupling block i to block j | **0** |
| state carried / slots occupied | 12 / 144 (**12× redundancy**) |

The algebra is ℍ⊕ℍ⊕ℍ. The three quaternions evolve side by side and have no
channel through which to exchange anything, so the node is three independent 4D
systems in a container twelve times larger than their state. Whatever "unified
simulation of physical fields" means, it is not this: a coupled electromagnetic–
thermal–mechanical voxel is precisely a system in which perturbing one part moves
the others, and the measured off-block response is identically zero.

The number 12 is also derived twice, incompatibly, inside the same paper. §2.1
builds it as 3 quaternions × 4 components. §6 directs development toward "the
fundamental gauge group U(1) × SU(2) × SU(3)" and §2.3 speaks of "the macroscopic
12D gauge groups"; dim U(1) + dim SU(2) + dim SU(3) = 1 + 3 + 8 = 12. These are
different twelves — three copies of a 4-dimensional algebra versus the Lie algebra
of a 12-dimensional group, which has no quaternionic block structure — and nothing
in the paper connects them.

The number 112 is given no derivation at all. §2.3 says it "is derived from the
necessary degrees of freedom to encapsulate the macroscopic 12D gauge groups
alongside the microstate variables of crystallography, spin configurations, and
valence orbital interactions required for phase-transition dynamics", without a
single count. 112 = 28 × 4 = 14 × 8; the paper mentions neither factorisation, so
there is nothing to audit.

## 4. σ(W·S + b) is not a symplectic integrator

§2.3 is the paper's strongest physical claim: the engine "employs a symplectic
integration scheme directly within the matrix multiplication", the update is

    S(t+1) = σ( W_PINN ⊗ S(t) + b )

and this "preserves the Lie group symmetries of the 12D/112D space, ensuring that
energy and momentum remain invariant without requiring floating-point correction
loops". `symplectic.py` tests it three ways.

**Isolate σ and b.** Take a harmonic oscillator and choose W = exp(dt·J), which is
exactly symplectic, so nothing is left to blame on a bad fit. 10⁵ steps,
dt = 0.01, E(0) = 0.5:

| update rule | E(end) | E/E(0) |
|---|---|---|
| leapfrog, no training | 0.499991 | 1.00 |
| W ∈ Sp(2), σ = id, b = 0 | 0.500000 | 1.00 |
| W ∈ Sp(2), σ = id, **b = 10⁻³** | 0.547046 | **1.09** |
| W ∈ Sp(2), **σ = tanh**, b = 0 | 1.00·10⁻⁵ | **2·10⁻⁵** |
| W ∈ Sp(2), **σ = relu**, b = 0 | 2.27·10⁻⁵ | **4.5·10⁻⁵** |

The only conserving row is the one where σ is the identity and b is zero — i.e.
where the update is a linear symplectic map and not a neural network layer. The
nonlinearity the paper specifies costs five orders of magnitude of energy; a bias
of 10⁻³ gains 9 %.

**Count the constraints.** Symplecticity is WᵀJW = J, a closed condition of
codimension dim GL − dim Sp:

| 2n | dim GL(2n) | dim Sp(2n) | constraints | fraction |
|---|---|---|---|---|
| 12 | 144 | 78 | 66 | 45.8 % |
| 112 | 12544 | 6328 | **6216** | **49.6 %** |

At 112×112, roughly half the weight matrix must satisfy equalities the paper never
writes down, and nothing in a PINN trajectory loss imposes them. A random 112×112
W misses the condition by max |WᵀJW − J| = 1.19. Sp(112) is measure zero in
GL(112), so "preserves the Lie group symmetries" is a property that a learned
W has with probability zero.

**Train it the way the paper proposes, then measure.** Fit W and b by least squares
against 20 000 exact one-step pairs from a true Hamiltonian flow, then roll out
10⁵ steps:

| system | rule | 1-step error | \|WᵀJW − J\| | E(10⁵) | E/E(0) |
|---|---|---|---|---|---|
| harmonic | leapfrog, no training | — | — | 1.12498 | **1.0000** |
| harmonic | σ = id | 4.4·10⁻¹⁶ | 3.3·10⁻¹⁶ | 1.12498 | **1.0000** |
| harmonic | σ = tanh | 4.4·10⁻² | 2.7·10⁻¹ | 2.73851 | **2.4342** |
| pendulum | leapfrog, no training | — | — | 1.41612 | **1.0000** |
| pendulum | σ = id | 1.5·10⁻³ | 2.5·10⁻⁵ | 14.4747 | **10.2212** |
| pendulum | σ = tanh | 4.4·10⁻² | 2.7·10⁻¹ | 2.46981 | **1.7440** |

For the harmonic oscillator the exact one-step map *is* linear and symplectic, so
least squares recovers it to 4·10⁻¹⁶ and energy is conserved. That is the special
case, not the architecture working — and adding the nonlinearity the paper
specifies makes the same system gain a factor 2.4. For the pendulum nothing
conserves: the linear fit is accurate to 1.5·10⁻³ after a single step, which is a
good fit, and ends ten times too energetic. Leapfrog holds both to five decimal
places with no weights and no training.

A small one-step residual is not conservation. Conservation is a property of the
structure of the update map, and σ(W·x + b) does not have that structure for any W
and b. The sentence "without requiring floating-point correction loops" has it
backwards: the correction loops are what a non-symplectic scheme needs, and the
proposed scheme is non-symplectic.

A note on notation: ⊗ in that equation cannot be a tensor product. The Kronecker
product of two 112×112 matrices has 157 351 936 entries, 300 MiB per node in fp16.
Read as the matrix product the paper means, one node's W·S is 2 809 856 FLOP; at
the RTX 3090's 142 TFLOP/s fp16 and 1.70 GHz boost the whole chip delivers 83 529
FLOP per cycle, so a single node takes **33.6 cycles of the entire GPU**, not "a
single clock cycle".

## 5. The block matrix is 112× overhead at the same arithmetic intensity

This is the quiet defect, and it is the one that decides whether the architecture
could work at all. The physics in an openOL node is 112 numbers — the paper's own
list of potentials, spins, lattice and valence variables. The node stores them as a
112×112 matrix:

| | |
|---|---|
| 112×112 fp16 block matrix | 24.5 KiB |
| the 112 physical potentials | 0.219 KiB |
| overhead | **112×** |
| nonzero fraction if built from 28 quaternion 4×4 blocks | 3.57 % |

The justification offered for paying that is tensor-core throughput: matrices are
"a format natively processed by Tensor Cores". But the roofline position is
unchanged by the representation:

| operation | FLOP | bytes | FLOP/byte |
|---|---|---|---|
| W·S, 112×112 node | 2 809 856 | 50 176 | **56.0** |
| W·s, 112-vector | 25 088 | 448 | **56.0** |

Identical arithmetic intensity, 112× the FLOPs and 112× the traffic. Both sit
*below* the RTX 3090 ridge point of 152 FLOP/byte, so both are memory bound, and
promoting the state to a matrix moves nothing toward the compute-bound regime
where tensor cores pay off. It multiplies the work by 112 and leaves the
bottleneck exactly where it was.

This also explains §3's own number. The paper reports that the pipeline "keeps
Tensor Core load in the efficient 15-20 % range". At 142 TFLOP/s peak that is
21.3–28.4 TFLOP/s delivered and 113.6–120.7 TFLOP/s idle. Fifteen to twenty per
cent utilisation of the one unit the entire architecture is built around is the
symptom, not a design target: §2 moves all physics onto the tensor cores precisely
so that they saturate. The measurement above gives the reason they do not — at 56
FLOP/byte the node waits on VRAM however the work is scheduled.

## 6. Memory and the step rate

Double-buffered state only — no weight matrices, no framebuffer, no rendering —
against the RTX 3090's 936 GB/s and 24 GB:

| grid | nodes | S, S′ | × 24 GB | cards | s/step | Hz |
|---|---|---|---|---|---|---|
| 32³ | 32 768 | 1.5 GiB | 0.1× | 1 | 0.002 | 569 |
| 64³ | 262 144 | 12.2 GiB | 0.5× | 1 | 0.014 | 71 |
| 80³ | 512 000 | 23.9 GiB | 1.0× | 1 | 0.027 | 36 |
| 100³ | 1 000 000 | 46.7 GiB | 1.9× | 2 | 0.054 | **19** |
| 128³ | 2 097 152 | 98.0 GiB | 4.1× | 5 | 0.112 | 8.9 |
| 256³ | 16 777 216 | 784.0 GiB | 32.7× | 33 | 0.899 | 1.1 |

Nodes whose state alone fills 24 GB: **513 588** (80³). Nodes the bandwidth allows
at 60 Hz: **310 906** (68³). So §3's "millions of nodes" costs 47 GiB per million
for state alone, and the real-time target of §5's "photorealistic synthetic
worlds" is a 68-voxel cube with nothing drawn.

§5 also proposes "searching for superconductors and novel alloy phase structures
in the full 112D space of openOL via molecular brute-force simulation". At one node
per atom, 24 GB holds 513 588 atoms — a **21.7 nm** cube of crystalline silicon,
state only. That is a legitimate size for atomistic work, which is exactly why the
comparison is unfavourable: classical molecular dynamics reaches that scale today
at a few tens of bytes per atom rather than 24.5 KiB, and without a 112-dimensional
claim. One mole of nodes would need 1.2·10¹⁸ cards.

## 7. Zero-branching is contradicted by its own section

The "zero-branching" property is the paper's stated engineering contribution:
"Solids, liquids, and electrical currents are computed by the identical hardware
logic without any if/else statements." Two paragraphs later, in the Projection
Phase: "Movement through empty space is rapidly handled by CUDA cores; upon
intersection with matter, the computation of refraction, reflection, or
interference is handed back to the Tensor Cores."

That is a branch. It is a per-ray data-dependent branch, which on a GPU is the
expensive kind: threads in a warp that hit matter at different depths diverge, and
the warp pays for every path taken. Raymarching into a sparse volume is one of the
more divergent workloads in graphics. The architecture does not remove branching;
it moves it from the physics kernel to the ray loop, where it costs more.

## 8. Hysteresis is a material property, not a shape

§5 proposes "replacing traditional BLDC motor design with evolutionary iteration in
openQL", producing "optimized, organically shaped 3D fractal models of stators and
rotors that eliminate hysteresis losses".

Hysteresis loss per unit volume per cycle is the area enclosed by the material's
B–H loop, and the total is that area times frequency times core volume. Geometry
enters only through the flux density the material is driven to and through the
volume of material present, so shape optimisation can reduce hysteresis loss and
is routinely used to do so. Eliminating it requires changing the material — a
lower-coercivity alloy, an amorphous or nanocrystalline ribbon, ferrite — or
removing the ferromagnetic core, as in an ironless or air-core machine, which
trades the loss for a much larger magnetising current. No shape, fractal or
otherwise, sets the B–H loop area to zero. The same conflation of a material
property with a computable geometry appeared in the MPPT preprint's hysteresis
argument.

## 9. Naming and presentation

Two factual notes, neither of which bears on the physics.

**openQL is an existing project.** OpenQL is a quantum programming framework from
QuTech / TU Delft, published in ACM Transactions on Quantum Computing
([10.1145/3474222](https://doi.org/10.1145/3474222), preprint
[arXiv:2005.13283](https://arxiv.org/abs/2005.13283)), with public documentation
and repository. The name is in use for a different kind of compiler in an adjacent
field.

**The author template was not completed.** The deposited PDF's header reads
`Author: Michal Mazgal [ORCID: enter-your-orcid]`. The placeholder is in the
published record.

**What the references carry.** All five exist, and four are cited for claims they
can support: Kuipers for the quaternion matrix, Raissi et al. for encoding physics
in network weights, Markidis et al. for MMA throughput, Mildenhall et al. for
volumetric rendering without polygons. The fifth, Baez, is cited for 12D/112D
mappings that are not in it, and is the text that excludes §2's central operation
(see §2). Citing real sources is a change from the earlier preprints; citing them
for what they contain is the step still outstanding.

## 10. What changed

Relative to the earlier preprints from this author, two things improved and one
did not.

Improved: a bibliography exists for the first time, and one central algebraic
step — the 4×4 representation of ℍ — is both correct and completely carried out.
In the September trilogy the algebra was right but no derivation was finished; in
MPPT the octonion mapping was not checkable at all. §1 of this evaluation is longer
than the corresponding section of the previous two.

Unchanged: the structure of the argument. A correct algebraic fact is stated, a
second one is attached to it that is excluded by a theorem, and hardware claims are
layered on top that fail on a single division. In MPPT the correct part was the
Fano plane and the excluded part was labelling basis units with physical
dimensions; here the correct part is the quaternion matrix and the excluded part is
a matrix form for 𝕆.

One detail is worth recording for its own sake. This is the first of the author's
deposits to attach Maxwell to quaternions rather than to octonions: §2.1 describes
the 4D layer as "quaternion mechanics (for computing phase shifts, rotations, and
Maxwell's equations)". Of the nine deposits, the three engineering preprints
(MPPT v1.0 and v1.1, the acoustic projector, X-Ternary) credit Maxwell with an
"original hypercomplex architecture" built on octonions, and the trilogy does not
mention Maxwell at all. Quaternions are the algebra for which the historical claim
has any purchase — §§618–619 of the 1873 *Treatise*, two pages of Hamiltonian
operator notation in which Maxwell never multiplies two quaternions — and
octonions appear nowhere in Maxwell. The attribution here is still uncited and
still carries no mechanism, but it is pointed at the right algebra for the first
time.

## 11. Limitations

This evaluation is of a 7-page conceptual whitepaper with no accompanying code, no
equations beyond those quoted, and no numerical results. There is nothing to run
against the paper's own implementation, because there is none, so §§5–6 are
arithmetic on published hardware specifications rather than benchmarks. The
reference card is the RTX 3090 named in this author's other preprints; on an H100
(3.35 TB/s, 990 TFLOP/s fp16) the ridge point rises to 295 FLOP/byte, the node
remains memory bound at 56 FLOP/byte, and the figures of §6 scale with the card:
resident nodes by 80/24 = 3.3× and the step rate by 3.35/0.936 = 3.6×. The
conclusions of §§5–6 do not depend on which card is assumed.

§3 reads "concatenating three such state matrices" as block-diagonal assembly.
That is the only reading that yields a 12×12 object from three 4×4 matrices, and
the 12D layer requires a 12×12 object, but the paper does not write the assembly
down. Were some other, coupling assembly intended, the result would not be a
representation of ℍ⊕ℍ⊕ℍ — and would also not be a representation of any
composition algebra, since Hurwitz's theorem leaves no 12-dimensional one.

§2's theorem concerns homomorphisms of algebras. Octonions can of course be
*stored* in matrices, and octonion multiplication can be *computed* on a GPU by
contracting an 8×8×8 structure-constant tensor, which is what this author's own
MPPT code does. What cannot exist is a matrix form in which matrix multiplication
performs octonion multiplication, which is what "transforming octonion algebra
into unified block matrices" so that the result is "natively processed by Tensor
Cores" requires.

## 12. Data availability

All numbers in this write-up are produced by `./run_all.sh` in this repository:
`quaternion.py` (§§1, 3), `octonion.py` (§2), `symplectic.py` (§4), `roofline.py`
(§§5, 6). Pure CPU, no network, about 20 seconds. `fano.py` is the multiplication
table from the companion `causal-trilogy-eval` repository, unmodified, so §2
measures the author's own structure constants. Full output of one run:
[`results.log`](results.log).

## References

- M. Mazgal, *openQL / openOL: Tensor Matrix Architecture for Unified Simulation of Physical Fields in 12D and 112D via GPU Tensor Cores*, Zenodo, 2026. DOI [10.5281/zenodo.23112560](https://doi.org/10.5281/zenodo.23112560)
- J. C. Baez, *The Octonions*, Bulletin of the AMS **39** (2002) 145–205. DOI [10.1090/S0273-0979-01-00934-X](https://doi.org/10.1090/S0273-0979-01-00934-X)
- A. Hurwitz, *Über die Composition der quadratischen Formen von beliebig vielen Variabeln*, Nachr. Ges. Wiss. Göttingen (1898) 309–316
- J. B. Kuipers, *Quaternions and Rotation Sequences*, Princeton University Press, 1999
- M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. **378** (2019) 686–707. DOI [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2nd ed., Springer, 2006
- N. Khammassi, I. Ashraf, J. van Someren, R. Nane, A. M. Krol, M. A. Rol, L. Lao, K. Bertels, C. G. Almudever, *OpenQL: A Portable Quantum Programming Framework for Quantum Accelerators*, ACM Trans. Quantum Computing **3** (2022). DOI [10.1145/3474222](https://doi.org/10.1145/3474222)
- S. Williams, A. Waterman, D. Patterson, *Roofline: an insightful visual performance model*, Comm. ACM **52** (2009) 65–76. DOI [10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
- J. C. Maxwell, *A Treatise on Electricity and Magnetism*, Clarendon Press, 1873, §§618–619
