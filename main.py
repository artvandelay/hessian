import sys
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import block_partitioned_matrices as bpm
from hessian import SequenceOfDenseBlocks


def make_model(num_layers: int, input_dim: int, hidden_dim: int, num_classes: int):
    return SequenceOfDenseBlocks(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        activation=torch.tanh,
    )


def random_parameter_vector(model: SequenceOfDenseBlocks) -> bpm.Vertical:
    return bpm.Vertical(
        [
            torch.randn(sum(p.numel() for p in layer.parameters()), 1)
            for layer in model
        ]
    )


def zeros_like_vertical(v: bpm.Vertical) -> bpm.Vertical:
    return bpm.Vertical([torch.zeros_like(block) for block in v.flat])


def dot_vertical(a: bpm.Vertical, b: bpm.Vertical) -> torch.Tensor:
    return torch.dot(a.to_tensor().flatten(), b.to_tensor().flatten())


def cg_solve(matvec, b: bpm.Vertical, tol: float = 1e-8, max_iter: int = 200):
    x = zeros_like_vertical(b)
    r = b - matvec(x)
    p = r
    rsold = dot_vertical(r, r)
    for _ in range(max_iter):
        ap = matvec(p)
        alpha = rsold / dot_vertical(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = dot_vertical(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def correctness_vs_cg(model, z_in, target, epsilon):
    b = random_parameter_vector(model)

    def matvec(v):
        return model.hessian_vector_product(z_in, target, v) + epsilon * v

    x_rahimi = model.hessian_inverse_product(z_in, target, b, epsilon)
    x_cg = cg_solve(matvec, b)

    rel_err = (x_rahimi.to_tensor() - x_cg.to_tensor()).norm() / x_cg.to_tensor().norm()
    return rel_err.item()


def scaling_benchmark(depths, input_dim, hidden_dim, num_classes, epsilon):
    results = []
    for num_layers in depths:
        model = make_model(num_layers, input_dim, hidden_dim, num_classes)
        z_in = torch.randn(1, input_dim, requires_grad=True)
        target = torch.randint(0, num_classes, (1,))
        b = random_parameter_vector(model)

        start = time.perf_counter()
        _ = model.hessian_inverse_product(z_in, target, b, epsilon)
        elapsed = time.perf_counter() - start
        results.append((num_layers, elapsed))
    return results


def stability_sweep(model, z_in, target, epsilons):
    b = random_parameter_vector(model)
    results = []
    for eps in epsilons:
        x = model.hessian_inverse_product(z_in, target, b, eps)
        x_tensor = x.to_tensor()
        finite = torch.isfinite(x_tensor).all().item()
        results.append((eps, finite, x_tensor.norm().item()))
    return results


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)

    input_dim = 2
    hidden_dim = 2
    num_classes = 2

    print("=== Correctness vs CG ===")
    model = make_model(num_layers=20, input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    z_in = torch.randn(1, input_dim, requires_grad=True)
    target = torch.randint(0, num_classes, (1,))
    epsilon = 1e-2
    rel_err = correctness_vs_cg(model, z_in, target, epsilon)
    print(f"relative error: {rel_err:.3e}")

    print("\n=== Scaling benchmark ===")
    depths = [20, 40, 80, 160, 320]
    scaling = scaling_benchmark(depths, input_dim, hidden_dim, num_classes, epsilon)
    for depth, elapsed in scaling:
        print(f"L={depth:<4d} time={elapsed:.4f}s")

    print("\n=== Stability sweep ===")
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    stability = stability_sweep(model, z_in, target, epsilons)
    for eps, finite, norm in stability:
        print(f"eps={eps:.0e} finite={finite} norm={norm:.3e}")


if __name__ == "__main__":
    main()
