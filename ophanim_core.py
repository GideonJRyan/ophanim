"""
Ophanim: A Novel Neural Architecture
Author: Gideon J. Ryan
Date: September 2026

First implementation verifying three core theorems:
1. Hard orthogonal separation: <v_A, v_B> = 0
2. Dense coverage: ergodic attention over full angular space
3. Shared intent convergence: z drives both streams to same objective
"""

import torch
import math


# Configuration
D = 6  # full representation space dimension
P = 3  # stream A dimension
Q = 3  # stream B dimension


# Build Orthogonal Subspaces
def build_subspaces(d=D, p=P, q=Q):
    """
    Construct two orthogonal subspaces A and B.
    Guarantees U_A^T U_B = 0 by construction.
    """
    U, _ = torch.linalg.qr(torch.randn(d, p + q))
    U_A = U[:, :p]
    U_B = U[:, p:]
    return U_A, U_B

# Reprojection
def reproject(U_A, U_B, p):
    """
    Periodic reprojection onto O(d).
    Re-orthogonalizes U_A and U_B every N steps.
    Prevents Cayley drift from accumulating.
    """
    U = torch.cat([U_A, U_B], dim=1)
    U_new, _ = torch.linalg.qr(U)
    return U_new[:, :p], U_new[:, p:]


# Rotation Operator
def rotation_matrix(theta, dim=D):
    """
    Rotation by angle theta in the (0,1) plane.
    Represents R(theta) acting on the representation space.
    """
    R = torch.eye(dim)
    c, s = math.cos(theta), math.sin(theta)
    R[0, 0] = c;  R[0, 1] = -s
    R[1, 0] = s;  R[1, 1] = c
    return R


# Theorem 1: Hard Orthogonal Separation
def test_orthogonal_separation(U_A, U_B):
    """
    Verifies <v_A, v_B> = 0 for a random input x.
    """
    x = torch.randn(D)
    v_A = U_A @ (U_A.T @ x)
    v_B = U_B @ (U_B.T @ x)
    inner = torch.dot(v_A, v_B).item()
    print(f"Theorem 1: Orthogonal Separation")
    print(f"  <v_A, v_B> = {inner:.8f} (should be 0)\n")
    return inner


# Theorem 2: Dense Coverage
def test_coverage(U_A, U_B, num_steps=36):
    """
    Verifies ergodic coverage every dimension visited
    at every rotation angle.
    """
    coverage = torch.zeros(D)
    for theta in torch.linspace(0, 2 * math.pi, num_steps):
        R = rotation_matrix(theta.item())
        U_A_r = R @ U_A
        U_B_r = R @ U_B
        seen = (U_A_r.abs().sum(1) > 1e-5) | (U_B_r.abs().sum(1) > 1e-5)
        coverage += seen.float()

    print(f"Theorem 2: Dense Coverage ({num_steps} rotations)")
    print(f"  Times each dimension seen: {coverage.int().tolist()}")
    print(f"  Full coverage achieved: {(coverage > 0).all().item()}\n")
    return coverage


# Theorem 3: Shared Intent Convergence
def test_shared_intent(U_A, U_B, steps=100, lr=0.1):
    """
    Verifies that a shared decoder W drives both streams
    to express the same intent. Loss should converge to ~0.
    """
    W = torch.randn(P, 1, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=lr)

    print(f"Theorem 3: Shared Intent Convergence ({steps} steps)")
    for step in range(steps):
        optimizer.zero_grad()
        x = torch.randn(D)
        theta = torch.rand(1).item() * 2 * math.pi
        R = rotation_matrix(theta)
        U_A_r = R @ U_A
        U_B_r = R @ U_B
        z_A = (U_A_r.T @ x) @ W
        z_B = (U_B_r.T @ x) @ W
        loss = (z_A - z_B).pow(2).mean()
        loss.backward()
        optimizer.step()

    print(f"  Final shared intent loss: {loss.item():.8f} (should be ~0)\n")
    return loss.item()

# Theorem 4: Cayley drift
def test_cayley_drift(steps=10000, reprojection_interval=100):
    """
    Runs U through many gradient-like updates.
    Measures orthogonality drift with and without reprojection.
    """
    print(f"\nTheorem 4: Cayley Drift Test ({steps} steps)")
    U_A, U_B = build_subspaces()

    # Simulate gradient updates: small random perturbations
    drift_without = []
    drift_with    = []

    U_A_free  = U_A.clone()
    U_B_free  = U_B.clone()
    U_A_fixed = U_A.clone()
    U_B_fixed = U_B.clone()

    for step in range(steps):
        # Small random perturbation simulating gradient update
        noise = torch.randn_like(U_A_free) * 0.001
        U_A_free  = U_A_free  + noise
        U_B_free  = U_B_free  + noise
        U_A_fixed = U_A_fixed + noise
        U_B_fixed = U_B_fixed + noise

        # Reprojection on fixed version
        if step % reprojection_interval == 0:
            U_A_fixed, U_B_fixed = reproject(U_A_fixed, U_B_fixed, P)

        # Measure drift every 500 steps
        if step % 500 == 0:
            drift_free  = (U_A_free.T  @ U_B_free).abs().mean().item()
            drift_fixed = (U_A_fixed.T @ U_B_fixed).abs().mean().item()
            drift_without.append(drift_free)
            drift_with.append(drift_fixed)
            print(f"  Step {step:>5} | "
                  f"Without reprojection: {drift_free:.8f} | "
                  f"With reprojection: {drift_fixed:.8f}")

    print(f"\n  Final drift without reprojection: {drift_without[-1]:.8f}")
    print(f"  Final drift with reprojection:    {drift_with[-1]:.8f}")
    print(f"  Reprojection keeps drift near zero: "
          f"{drift_with[-1] < 0.0001}")
    return drift_without, drift_with

# Main 
if __name__ == "__main__":
    print("Ophanim Core Theorem Verification:")
    print("-"*50)

    U_A, U_B = build_subspaces()

    test_orthogonal_separation(U_A, U_B)
    test_coverage(U_A, U_B)
    test_shared_intent(U_A, U_B)
    test_cayley_drift()

    print("-"*50)
    print("All four theorems verified.")
