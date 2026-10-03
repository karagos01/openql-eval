#!/usr/bin/env python3
"""Does S(t+1) = sigma(W_PINN (x) S(t) + b) conserve energy?

openQL Sec. 2.3 states that the engine "employs a symplectic integration scheme
directly within the matrix multiplication", that the update is

    S(t+1) = sigma( W_PINN (x) S(t) + b )

where W_PINN holds the physical laws learned by a physics-informed network, and
that this "preserves the Lie group symmetries of the 12D/112D space, ensuring
that energy and momentum remain invariant without requiring floating-point
correction loops".

Three measurements:

  1. Roll the update out on a harmonic oscillator with W chosen to be exactly
     symplectic, and vary only sigma and b.  This isolates the cost of the
     nonlinearity and the bias from any question of whether W was learned well.

  2. Count the constraints.  Being symplectic is a closed condition of
     codimension dim GL - dim Sp on the weight matrix.  Nothing in the paper
     imposes it, and no term in a PINN loss on trajectory data imposes it either.

  3. Train W and b the way the paper proposes -- least squares against exact
     one-step data from a real Hamiltonian system -- and then measure the energy
     after a long rollout, against leapfrog, which conserves energy with no
     training at all.  A small one-step residual is not conservation.
"""
import numpy as np

np.random.seed(0)

# ---------------------------------------------------------------- 1. sigma, b
dt = 0.01
N = 100_000
H1 = lambda q, p: 0.5 * (p * p + q * q)
W_sym = np.array([[np.cos(dt), np.sin(dt)],         # exp(dt J), exactly symplectic
                  [-np.sin(dt), np.cos(dt)]])


def leapfrog1(n=N, q=1.0, p=0.0):
    for _ in range(n):
        p -= 0.5 * dt * q
        q += dt * p
        p -= 0.5 * dt * q
    return H1(q, p)


def rollout(sigma=lambda x: x, b=0.0, W=W_sym, n=N, x0=(1.0, 0.0)):
    x = np.array(x0)
    for _ in range(n):
        x = sigma(W @ x + b)
    return H1(x[0], x[1])


E0 = H1(1.0, 0.0)
cases = [("leapfrog, no training", leapfrog1()),
         ("W in Sp(2), sigma = id,   b = 0", rollout()),
         ("W in Sp(2), sigma = id,   b = 1e-3", rollout(b=1e-3)),
         ("W in Sp(2), sigma = tanh, b = 0", rollout(sigma=np.tanh)),
         ("W in Sp(2), sigma = relu, b = 0",
          rollout(sigma=lambda x: np.maximum(x, 0.0)))]
print(f"1. Harmonic oscillator, {N:,} steps, dt = {dt}, E(0) = {E0:.6f}")
print(f"   {'update rule':<36s} {'E(end)':>13s} {'E/E0':>12s}")
for name, E in cases:
    print(f"   {name:<36s} {E:>13.6g} {E/E0:>12.3g}")
print("   -> the only row that conserves is the one where sigma is the identity")
print("      and b is zero, i.e. where the update is a linear symplectic map and")
print("      not a neural network layer at all.  tanh loses five orders of")
print("      magnitude of energy; a bias of 1e-3 gains 9 %.")

# ------------------------------------------------------------ 2. codimension
print("\n2. How large is the symplectic condition?")
print(f"   {'2n':>5s} {'dim GL(2n)':>11s} {'dim Sp(2n)':>11s} {'constraints':>12s}"
      f" {'fraction':>9s}")
for twon in (12, 112):
    n = twon // 2
    gl, sp = twon * twon, n * (2 * n + 1)
    print(f"   {twon:>5d} {gl:>11d} {sp:>11d} {gl-sp:>12d} {(gl-sp)/gl:>8.1%}")
n = 56
J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
W = np.random.randn(2 * n, 2 * n) / np.sqrt(2 * n)
print(f"   a random 112x112 W misses the condition by max |W^T J W - J| ="
      f" {np.abs(W.T @ J @ W - J).max():.3f}")
print("   -> at 112x112, 6216 of 12544 degrees of freedom must satisfy equalities")
print("      the paper never writes down.  Sp(112) is measure zero in GL(112).")

# -------------------------------------------- 3. train it, then measure energy
print("\n3. Train W and b on exact data, then roll out")

s = 3.0                                   # keep the scaled state inside (-1, 1)
Hp = lambda th, om: 0.5 * om * om + (1.0 - np.cos(th))


def step_pendulum(x):                     # leapfrog step, scaled coordinates
    th, om = x[..., 0] * s, x[..., 1] * s
    om = om - 0.5 * dt * np.sin(th)
    th = th + dt * om
    om = om - 0.5 * dt * np.sin(th)
    return np.stack([th / s, om / s], axis=-1)


def step_harmonic(x):
    q, p = x[..., 0] * s, x[..., 1] * s
    p = p - 0.5 * dt * q
    q = q + dt * p
    p = p - 0.5 * dt * q
    return np.stack([q / s, p / s], axis=-1)


def fit(step, sigma):
    """Least squares for W, b in x' = sigma(W x + b) on 20 000 exact pairs."""
    X = np.random.uniform(-0.7, 0.7, size=(20_000, 2))
    Y = step(X)
    T = np.arctanh(np.clip(Y, -0.999999, 0.999999)) if sigma == "tanh" else Y
    A = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(A, T, rcond=None)
    Wf, bf = sol[:2].T, sol[2]
    pred = np.tanh(X @ Wf.T + bf) if sigma == "tanh" else X @ Wf.T + bf
    return Wf, bf, np.abs(pred - Y).max()


def roll(Wf, bf, sigma, n, x0):
    x = np.array(x0)
    for _ in range(n):
        y = Wf @ x + bf
        x = np.tanh(y) if sigma == "tanh" else y
    return x


for label, step, Hf, x0 in (("harmonic (linear)", step_harmonic, H1, (0.5, 0.0)),
                            ("pendulum (nonlinear)", step_pendulum, Hp, (2.0/s, 0.0))):
    E0 = Hf(x0[0] * s, x0[1] * s)
    print(f"\n   {label}, E(0) = {E0:.6f}")
    print(f"     {'rule':<26s} {'1-step err':>11s} {'|W^T J W - J|':>14s}"
          f" {'E(1e5)':>12s} {'E/E0':>10s}")
    x = np.array(x0)
    for _ in range(N):
        x = step(x)
    E = Hf(x[0] * s, x[1] * s)
    print(f"     {'leapfrog, no training':<26s} {'-':>11s} {'-':>14s}"
          f" {E:>12.6g} {E/E0:>10.4f}")
    for sigma in ("id", "tanh"):
        Wf, bf, res = fit(step, sigma)
        J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])
        dev = np.abs(Wf.T @ J2 @ Wf - J2).max()
        xe = roll(Wf, bf, sigma, N, x0)
        E = Hf(xe[0] * s, xe[1] * s)
        print(f"     {'sigma(Wx+b), sigma = ' + sigma:<26s} {res:>11.2e}"
              f" {dev:>14.2e} {E:>12.6g} {E/E0:>10.4f}")

print("\n   -> for the harmonic oscillator the true one-step map IS linear and")
print("      symplectic, so least squares recovers it to 4e-16 and energy is")
print("      conserved.  That is the special case, not the architecture working:")
print("      add the nonlinearity the paper specifies and the same system gains a")
print("      factor 2.4.  For the pendulum nothing conserves.  The linear fit is")
print("      accurate to 1.5e-3 after a single step, which is a good fit, and")
print("      ends ten times too energetic; the tanh fit lands at 1.74 x.  Neither")
print("      number is a correction loop away from 1.0000.  Leapfrog, with no")
print("      weights and no training, holds both to 5 decimal places, because")
print("      conservation is a property of the structure of the update and")
print("      sigma(W x + b) does not have that structure for any W and b.")
