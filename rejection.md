## Review of "Neural Operator Surrogates for Fast NMR Spin Dynamics Simulation and Parameter Estimation"

**Summary:** The paper proposes using neural networks as surrogate models to accelerate NMR spin dynamics simulations, claiming 10^4× speedup over exact solvers for N=12 spins.

**Major Issues:**

1. **Insufficient Technical Detail**
   - The "multi-head dense neural network" architecture is completely unspecified. No layer counts, widths, activation functions, or training procedures are described.
   - No information on training set size, sampling strategy for Hamiltonian parameters, or generalization bounds.
   - The claim of "O(1) complexity" is misleading—the network still has computational cost that scales with its size and the number of time steps predicted.

2. **Weak Validation**
   - Training only on N ≤ 10 systems but claiming results for N = 12 is extrapolation without justification. Where is the out-of-distribution performance analysis?
   - Single test case for inverse problem (one J-coupling value). No systematic validation across parameter space.
   - No quantitative error metrics provided anywhere. What are the actual L2 errors, maximum deviations, or R² values?

3. **Trivial Problem Setup**
   - 1D chain with uniform nearest-neighbor coupling is the simplest possible topology. Real biomolecules have complex 3D connectivity with heterogeneous coupling strengths.
   - Only tracking three observables. Real NMR experiments measure entire 2D/3D spectra with hundreds of cross-peaks.
   - 300 time steps is short for many NMR experiments (millisecond to second timescales with microsecond resolution).

4. **Unfair Comparisons**
   - Comparing against "sparse matrix diagonalization" is a strawman. Modern methods use Krylov subspace techniques, Chebyshev propagation, or tensor networks that are far more efficient.
   - No comparison against established approximation methods (restricted state spaces, selective populations, etc.).
   - The "exponential scaling" claim ignores that sparse methods don't actually scale as 2^N in practice due to structural exploitation.

5. **Missing Critical Analysis**
   - No discussion of when the surrogate fails. What about highly entangled states or long-range correlations?
   - Conservation laws: Does the network preserve trace, hermiticity, positivity of the density matrix?
   - What happens outside the training distribution? How does error accumulate with time?

6. **Questionable Claims**
   - "Real-time adaptive NMR spectroscopy" requires sub-microsecond inference for actual pulse sequences. No latency measurements provided.
   - The inverse problem demonstration is unconvincing—recovering one scalar parameter from a synthetic target where you control the forward model proves nothing about practical utility.

**Minor Issues:**

- Figure quality is poor (low resolution, unclear labels)
- References are incomplete (missing page numbers, some are just arXiv preprints from 2020-2021)
- The paper reads more like a preliminary proof-of-concept than a complete scientific contribution
- No code availability statement or reproducibility information

**The Fundamental Problem:**

This work conflates two distinct challenges: (1) accelerating forward simulation, and (2) solving inverse problems. Neural surrogates might help with (2) by providing fast gradients, but the paper doesn't demonstrate this convincingly. For (1), there are already excellent approximation schemes in the NMR simulation community that this work ignores entirely.

The claimed speedup is achieved by learning a very specific, simple problem and then comparing against an inefficient exact solver. This is not a fair or meaningful benchmark.

**Verdict:** Reject. 

The paper needs substantial additional work: comprehensive benchmarking against state-of-the-art approximate methods, rigorous error analysis, validation on realistic molecular systems, detailed architectural specifications, and demonstration on actual experimental data rather than synthetic test cases. The current manuscript is too preliminary for publication in a serious venue.
