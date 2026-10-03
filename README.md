# openQL / openOL — independent evaluation

Reproducible evaluation of M. Mazgal, *openQL / openOL: Tensor Matrix Architecture
for Unified Simulation of Physical Fields in 12D and 112D via GPU Tensor Cores*
(Zenodo, 3 October 2026, DOI
[10.5281/zenodo.23112560](https://doi.org/10.5281/zenodo.23112560)).

The write-up is in [`PAPER.md`](PAPER.md), in Czech in [`PAPER.cs.md`](PAPER.cs.md).

**The quaternion half is correct and complete** — the 4×4 matrix in §2.1 is the
left regular representation of ℍ and satisfies M(q)M(r) = M(qr) to 1.8·10⁻¹⁵, and
this is the first central algebraic step in this author's preprints to survive a
check end to end. **The octonion half of the same sentence is excluded by a
theorem:** matrix multiplication is associative, so any homomorphism 𝕆 → Mₙ(ℝ)
kills every associator, and the associators of the author's own multiplication
table span the whole imaginary part, leaving 1 of 8 dimensions. The paper's only
octonion citation, Baez's survey, states the non-associativity on its first page.
Beyond that, the 12×12 node is block diagonal and so is three decoupled
quaternions, the proposed update σ(W·S + b) is not symplectic, and the block-matrix
representation sits at the same point on the roofline as the 112-component state it
encodes while costing 112× the traffic.

## Requirements

```
python3 -m pip install -r requirements.txt   # numpy only
```

No GPU, no network, no model downloads. §§5–6 of `PAPER.md` are arithmetic on
published hardware specifications, because the preprint ships no code.

## Reproducing the results

```
./run_all.sh          # everything, about 20 seconds on any CPU
```

| Script | What it computes | Time |
|---|---|---|
| `quaternion.py` | the §2.1 matrix as a representation of ℍ; what "concatenating three such state matrices" produces; the two derivations of 12 | 2 s |
| `octonion.py` | associators of the published table, the span they fill, and the measured failure of the same construction for 𝕆 | 8 s |
| `symplectic.py` | energy under σ(W·S + b) with σ and b isolated; codimension of Sp; W fitted to exact data, then rolled out | 8 s |
| `roofline.py` | representation overhead, arithmetic intensity, resident nodes, step rate, tensor-core utilisation, atoms per card | 1 s |

`fano.py` is the multiplication table from the companion `causal-trilogy-eval`
repository, unmodified, so `octonion.py` measures the author's own structure
constants. Full output of one run: [`results.log`](results.log).

## Key numbers

| what | published | measured / computed |
|---|---|---|
| §2.1 4×4 matrix is a representation of ℍ | "isomorphic" | **correct**, error 1.78·10⁻¹⁵ |
| basis triples with a non-zero associator | — | 168 of 512, spanning all **7** dimensions of Im 𝕆 |
| dimensions of 𝕆 surviving any matrix form | "unified block matrices" | **1 of 8 (12.5 %)**, so φ is the zero map |
| same construction, L_x L_y vs L_{xy} | — | ℍ: 1.8·10⁻¹⁵ — **𝕆: 29.5 (relative 2.76)** |
| 12×12 node, response outside the perturbed block | "unified state structure" | **0.000** — the algebra is ℍ⊕ℍ⊕ℍ |
| 12×12 node, state vs slots | — | 12 numbers in 144 slots, 96 structurally zero |
| where 12 comes from | — | derived twice, incompatibly (3×4 and dim U(1)×SU(2)×SU(3)) |
| where 112 comes from | "derived from the necessary degrees of freedom" | no count given anywhere |
| σ(W·S + b), harmonic, σ = tanh | "energy … invariant" | **2·10⁻⁵ × E(0)** after 10⁵ steps |
| σ(W·S + b), pendulum, W fitted to exact data | "energy … invariant" | 1-step error 1.5·10⁻³, **10.2 × E(0)** |
| leapfrog on the same two systems, untrained | — | **1.0000 × E(0)** on both |
| symplectic condition at 112×112 | "preserves the Lie group symmetries" | **6216 of 12544** constraints, never written down |
| "tensor product ⊗ … in a single clock cycle" | 1 cycle | **33.6 cycles of the entire GPU**, per node |
| block matrix vs the state it encodes | "natively processed by Tensor Cores" | same **56 FLOP/byte**, 112× the FLOPs and traffic |
| nodes of state resident in 24 GB | "millions of nodes" | **513 588** (80³), state only |
| nodes the bandwidth allows at 60 Hz | "absolute real-time" | **310 906** (68³), nothing drawn |
| Tensor Core load | "efficient 15-20 % range" | 113.6–120.7 of 142 TFLOP/s **idle** |
| "molecular brute-force simulation" | full 112D space | 513 588 atoms = a **21.7 nm** silicon cube per card |
| "zero-branching" | no if/else | contradicted by §3's own ray/matter handoff |
| "eliminate hysteresis losses" via shape | eliminated | B–H loop area is a material property; shape reduces, never eliminates |

## Scope

The preprint is a 7-page conceptual whitepaper with no accompanying code and no
numerical results, so there is no implementation to benchmark. Two presentation
facts are recorded in §9 of the write-up rather than treated as defects: the
deposited PDF's header reads `Author: Michal Mazgal [ORCID: enter-your-orcid]`, and
OpenQL is already the name of a quantum programming framework from QuTech / TU
Delft ([10.1145/3474222](https://doi.org/10.1145/3474222)).

## Companion evaluations

- [`octonion-mppt-eval`](https://github.com/karagos01/octonion-mppt-eval) — the octonion two-layer/associator template and the magnon claims
- [`xternary-eval`](https://github.com/karagos01/xternary-eval) — the 2-bit LLM inference engine
- [`causal-trilogy-eval`](https://github.com/karagos01/causal-trilogy-eval) — the CQFT / PCTP / SOTP trilogy of September 2026

## Licence

Code (all `*.py` and `run_all.sh`): MIT, see `LICENSE`.
`PAPER.md` and `PAPER.cs.md`: CC BY 4.0.
