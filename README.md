# Hessian Inverse Product — Original Implementation Validation

This project validates the original implementation of the Hessian inverse product solver from:

**A. Rahimi — Fast Inversion of the Hessian of a Deep Network (2026)**

Paper: [https://arxiv.org/abs/2601.06096](https://arxiv.org/abs/2601.06096)
Repo: [https://github.com/a-rahimi/hessian](https://github.com/a-rahimi/hessian)

---

## Objective

To scientifically verify:

1. Numerical correctness of `(H + eps I)^-1 v` computation.
2. Linear scaling with network depth.
3. Agreement with conjugate-gradient approximations.
4. Stability behavior under damping.

This repository is treated as a **linear algebra solver**, not a training framework.

---

## Installation

```bash
git clone https://github.com/a-rahimi/hessian.git
cd hessian
pip install torch numpy scipy
```

---

## Minimal Sanity Test

Create `test_solver.py`:

```python
import torch
import time
from hessian import hessian_inverse_product
from examples import tall_skinny_network

torch.manual_seed(0)

L = 80
a = 1
p = 2

net = tall_skinny_network(L=L, a=a, p=p)
v = torch.randn(net.num_params)
eps = 1e-2

x = hessian_inverse_product(net, v, eps=eps)
print("Output norm:", x.norm().item())
```

Run:

```bash
python test_solver.py
```

Expected: finite non-NaN output.

---

## Correctness vs CG

Implement CG Hessian inversion and compare:

```python
# pseudo-code sketch
x_rahimi = hessian_inverse_product(net, v, eps)
x_cg = cg_solve(lambda z: net.hessian_vector_product(z)+eps*z, v)
print((x_rahimi-x_cg).norm()/x_cg.norm())
```

Expected relative error < 1e-4.

---

## Scaling Benchmark

Repeat solver for:

```python
L = [20,40,80,160,320]
```

Record runtime vs L.

Expected: approximately linear growth.

---

## What This Repo Is NOT

* Not a training optimizer.
* Not a stochastic Newton method.
* Not directly usable with minibatches.

It is a **deterministic Hessian linear system solver**.

---

## Why This Matters

This solver removes the main computational barrier to using second-order curvature in deep nets. The optimization design problem remains open.

---

## Research Usage Disclaimer

Newton-style steps require:

* Damping
* Trust regions
* Noise control

Using raw Hessian inverse in SGD will diverge.

---

## End Goal

To confidently state whether Rahimi’s algorithm:

> Computes Hessian inverse products correctly and efficiently in deep tall networks.

---
