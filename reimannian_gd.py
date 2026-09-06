"""
Ophanim: Riemannian Gradient Descent on O(d)
Closes Gap 3: replaces QR reprojection with proper
matrix exponential retraction.

From 06_Manifold_Constraint.md:
- Standard GD lives in flat Euclidean space
- O(d) is a curved manifold
- Valid steps must stay on the manifold surface
- Retraction via matrix exponential guarantees this exactly

Three methods compared:
1. Euclidean GD       — drifts off O(d)
2. QR Reprojection      — fast fix, approximate
3. Riemannian GD       — exact, mathematically honest
"""

import torch
import numpy as np
import math
import matplotlib.pyplot as plt

# Manifold Operations

def project_to_stiefel(U):
    """
    Projects U back onto O(d) via QR decomposition.
    Fast fix — Gap 1 solution.
    """
    Q, R = torch.linalg.qr(U)
    # Ensure positive diagonal of R (unique QR)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    return Q * signs.unsqueeze(0)


def riemannian_gradient(U, euclidean_grad):
    """
    Projects Euclidean gradient onto tangent space of O(d).

    Tangent space at U: {UΩ : Ω skew-symmetric}
    Riemannian gradient: G - U·sym(U^T·G)
    where sym(X) = (X + X^T) / 2
    """
    G = euclidean_grad
    UtG = U.T @ G
    sym_UtG = (UtG + UtG.T) / 2
    return G - U @ sym_UtG


def retract_exponential(U, riem_grad, lr):
    """
    Retracts along O(d) using matrix exponential.
    U(t+1) = U(t) · exp(-lr · U(t)^T · riem_grad)

    The matrix exponential of a skew-symmetric matrix
    is always orthogonal — stays on O(d) exactly.
    """
    A = -lr * U.T @ riem_grad
    # Skew-symmetrize A to ensure exact orthogonality
    A = (A - A.T) / 2
    # Matrix exponential via eigendecomposition
    # For skew-symmetric A: exp(A) is orthogonal
    exp_A = torch.linalg.matrix_exp(A)
    return U @ exp_A


def cayley_retract(U, riem_grad, lr):
    """
    Cayley transform approximation of matrix exponential.
    Cheaper than exact exp — O(d^2) vs O(d^3).

    exp(Ω) ≈ (I - η/2·Ω)(I + η/2·Ω)^{-1}
    """
    A = -lr * U.T @ riem_grad
    A = (A - A.T) / 2  # skew-symmetrize
    d = A.shape[0]
    I = torch.eye(d)
    cayley = torch.linalg.solve(I + lr/2 * A, I - lr/2 * A)
    return U @ cayley


# Loss Function On O(d)
def orthogonal_loss(U, target):
    """
    Simple loss: distance from U to a target orthogonal matrix.
    Minimum at U = target.
    """
    return ((U - target) ** 2).sum()


# Three Optimization Methods
def run_euclidean_gd(U_init, target, steps=200, lr=0.01):
    """Standard GD ignores manifold structure."""
    U = U_init.clone()
    losses = []
    orthogonality = []

    for step in range(steps):
        U.requires_grad_(True)
        loss = orthogonal_loss(U, target)
        loss.backward()
        with torch.no_grad():
            U = U - lr * U.grad
            U = U.detach()

        losses.append(loss.item())
        # Measure drift off O(d)
        drift = (U.T @ U - torch.eye(U.shape[0])).abs().mean().item()
        orthogonality.append(drift)

    return losses, orthogonality, U


def run_qr_reprojection(U_init, target, steps=200,
                         lr=0.01, interval=10):
    """QR reprojection every N steps — Gap 1 fast fix."""
    U = U_init.clone()
    losses = []
    orthogonality = []

    for step in range(steps):
        U.requires_grad_(True)
        loss = orthogonal_loss(U, target)
        loss.backward()
        with torch.no_grad():
            U = U - lr * U.grad
            U = U.detach()
            # Periodic reprojection
            if step % interval == 0:
                U = project_to_stiefel(U)

        losses.append(loss.item())
        drift = (U.T @ U - torch.eye(U.shape[0])).abs().mean().item()
        orthogonality.append(drift)

    return losses, orthogonality, U


def run_riemannian_gd(U_init, target, steps=200, lr=0.01):
    """
    Proper Riemannian GD with matrix exponential retraction.
    Stays on O(d) exactly at every step.
    """
    U = U_init.clone()
    losses = []
    orthogonality = []

    for step in range(steps):
        U.requires_grad_(True)
        loss = orthogonal_loss(U, target)
        loss.backward()

        with torch.no_grad():
            euc_grad = U.grad.clone()
            # Project gradient onto tangent space
            riem_grad = riemannian_gradient(U, euc_grad)
            # Retract via matrix exponential
            U = retract_exponential(U, riem_grad, lr)
            U = U.detach()

        losses.append(loss.item())
        drift = (U.T @ U - torch.eye(U.shape[0])).abs().mean().item()
        orthogonality.append(drift)

    return losses, orthogonality, U


# ── Verification ──────────────────────────────────────────────
def verify_riemannian_gd(d=8, steps=200, lr=0.01):
    print("=" * 60)
    print("Riemannian GD Verification — Gap 3")
    print("=" * 60)

    torch.manual_seed(42)

    # Random starting point on O(d)
    U_init, _ = torch.linalg.qr(torch.randn(d, d))

    # Random target on O(d)
    target, _ = torch.linalg.qr(torch.randn(d, d))

    print(f"\nDimension: {d}x{d}")
    print(f"Steps: {steps}, LR: {lr}")

    print("\n1. Euclidean GD...")
    l1, o1, U1 = run_euclidean_gd(
        U_init.clone(), target, steps, lr)

    print("2. QR Reprojection (every 10 steps)...")
    l2, o2, U2 = run_qr_reprojection(
        U_init.clone(), target, steps, lr)

    print("3. Riemannian GD (matrix exponential)...")
    l3, o3, U3 = run_riemannian_gd(
        U_init.clone(), target, steps, lr)

    # Summary
    print(f"\n{'-'*60}")
    print("SUMMARY")
    print(f"{'-'*60}")
    print(f"{'Method':<25} {'Final Loss':>12} "
          f"{'Final Drift':>13} {'On Manifold':>13}")
    print("-" * 65)

    for name, losses, ortho in [
        ("Euclidean GD",     l1, o1),
        ("QR Reprojection",  l2, o2),
        ("Riemannian GD",    l3, o3),
    ]:
        on_manifold = ortho[-1] < 1e-5
        print(f"{name:<25} {losses[-1]:>12.6f} "
              f"{ortho[-1]:>13.8f} {str(on_manifold):>13}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#E8593C', '#3B8BD4', '#1D9E75']
    labels = ['Euclidean GD', 'QR Reprojection', 'Riemannian GD']

    for losses, ortho, color, label in zip(
        [l1, l2, l3], [o1, o2, o3], colors, labels
    ):
        ax1.plot(losses, color=color, label=label, linewidth=1.5)
        ax2.plot(ortho,  color=color, label=label, linewidth=1.5)

    ax1.set_title('Loss Convergence', fontsize=13)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Orthogonality Drift ||U^TU - I||',
                  fontsize=13)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Drift (lower = better)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Ophanim: Riemannian GD vs Baselines',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('riemannian_gd.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved: riemannian_gd.png")

    # Key checks
    print("\nKey checks:")
    print(f"  Riemannian GD stays on manifold exactly: "
          f"{o3[-1] < 1e-6}")
    print(f"  Riemannian GD converges (loss < 0.1): "
          f"{l3[-1] < 0.1}")
    print(f"  Riemannian GD beats Euclidean GD drift: "
          f"{o3[-1] < o1[-1]}")


if __name__ == "__main__":
    verify_riemannian_gd()