# Structural Map — Hessian Inverse Product Solver

This map summarizes where the solver stages live in the codebase and the key
functions that implement each step.

## Core solver pipeline (high level)

1. **Derivative blocks (Dx, Dz, DD
yy)**
   * `SequenceOfBlocks.derivatives` computes per-layer mixed derivatives and
     returns them as block-diagonal matrices. It records per-layer `dloss_dout`
     via hooks, then calls each layer’s `derivatives` method.
   * `BlockWithMixedDerivatives.derivatives` builds Jacobians/Hessians using
     `torch.func` helpers and returns a `LayerDerivatives` tuple.
   * `LossLayer.derivatives` adapts the scalar-loss derivatives into 2D block
     shapes for consistent block algebra.

2. **Hessian–vector product (reference equation)**
   * `SequenceOfBlocks.hessian_vector_product` constructs the `M` and `P`
     matrices, validates shapes, and applies the explicit block-form of
     equation (H v) using block operations.

3. **Augmented system construction for (H + εI) x = b**
   * `SequenceOfBlocks.hessian_inverse_product` builds the 3×3 block matrix `K`
     that encodes the augmented system, with blocks composed of `DD_Dxx`,
     `DD_Dzx`, `Dx`, `M`, and `P`. This is the explicit augmented system
     described in `hessian.tex`.

4. **Permutation / pivoting to block‑tridiagonal form**
   * `Tridiagonal.blockwise_transpose` performs the permutation π that pivots
     the 3×3 block matrix `K` into a block‑tridiagonal matrix `K'`.
   * `Vertical.blockwise_transpose` performs the matching permutation for the
     right‑hand side vector `[b; 0; 0]`.
   * `SequenceOfBlocks.hessian_inverse_product` applies the permutation,
     solves, then pivots back using the same blockwise transpose.

5. **Block‑tridiagonal solve**
   * `Tridiagonal.solve` solves `K' xyz' = b'` by applying the block LDU
     factorization (`Tridiagonal.LDU_decomposition`) and forward/diagonal/back
     substitution (L.solve → D.solve → U.solve).
   * `Tridiagonal.LDU_decomposition` constructs the L, D, U factors via
     recurrence relations on the block diagonals and off‑diagonals.

6. **3×3 block solves inside the tridiagonal system**
   * `Generic3x3.solve` calls `_generic3x3_solve`, which computes a Schur
     complement and performs a structured LDU solve.
   * `_generic3x3_solve` includes explicit **forward substitution** and
     **backward substitution** steps:
     * Forward substitution: compute Z by solving L D Z = B.
     * Backward substitution: recover X by solving U X = Z.

## Stage‑to‑function map

| Stage | Primary implementation |
| --- | --- |
| Derivative block construction | `SequenceOfBlocks.derivatives`, `BlockWithMixedDerivatives.derivatives`, `LossLayer.derivatives` |
| Augmented system (K) | `SequenceOfBlocks.hessian_inverse_product` |
| Permutation (π K π) | `Tridiagonal.blockwise_transpose`, `Vertical.blockwise_transpose` |
| Block‑tridiagonal solve | `Tridiagonal.solve`, `Tridiagonal.LDU_decomposition` |
| 3×3 block solve | `Generic3x3.solve`, `_generic3x3_solve` |
| Forward/back substitution | `_generic3x3_solve` (explicit), `Tridiagonal.solve` (via L/D/U solves) |

## Notes

* The solver uses block‑structured matrix abstractions from
  `block_partitioned_matrices.py` to avoid materializing full Hessian blocks.
* The permutation is implemented as a **blockwise transpose**, which is an
  explicit structural rearrangement rather than a dense permutation matrix.
