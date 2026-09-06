"""
Ophanim: Toroidal Training Dynamics
Implements and verifies two fixes for toroidal loss landscape:

Fix 1 Langevin noise: escapes mandatory saddle points on T²
Fix 2 Winding penalty: prevents phase from orbiting indefinitely

From 07_Loss_Landscape.md:
- χ(T²) = 0 forces mandatory saddle points (Poincaré-Hopf)
- Winding number ≠ 0 means training runs but never converges
- Langevin + winding penalty together guarantee convergence

copy paste latex and symbols are used ffor better understanding
"""

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt


# Toroidal Phase Tracker
class ToroidalPhaseTracker:
    """
    Tracks (θ_A, θ_B) on T² = S¹ infinity S¹.
    Monitors winding numbers and convergence.
    """
    def __init__(self):
        self.theta_A = 0.0
        self.theta_B = 0.0
        self.history_A = [0.0]
        self.history_B = [0.0]

    def step(self, omega_A, omega_B):
        self.theta_A += omega_A
        self.theta_B += omega_B
        self.history_A.append(self.theta_A % (2 * math.pi))
        self.history_B.append(self.theta_B % (2 * math.pi))

    def winding_number(self):
        """
        Counts how many full rotations each phase has completed.
        Winding number ≠ 0 means orbiting - not converging.
        """
        wA = self.theta_A / (2 * math.pi)
        wB = self.theta_B / (2 * math.pi)
        return wA, wB


# Loss On Torus
def torus_loss(theta_A, theta_B):
    """
    A smooth loss function on T².
    Has one minimum, one maximum, two saddle points.
    Forced by Poincaré-Hopf: χ(T²) = 0.
    """
    a = math.cos(theta_A)
    b = math.cos(theta_B)
    c = math.cos(theta_A + theta_B)
    return -(0.5 * a + 0.5 * b + 0.3 * c)


def torus_grad(theta_A, theta_B, eps=1e-4):
    """Numerical gradient of loss on T²."""
    dA = (torus_loss(theta_A + eps, theta_B) -
          torus_loss(theta_A - eps, theta_B)) / (2 * eps)
    dB = (torus_loss(theta_A, theta_B + eps) -
          torus_loss(theta_A, theta_B - eps)) / (2 * eps)
    return dA, dB


# Standard Gradient Descent On Torus
def run_standard_gd(steps=500, lr=0.05, seed=42):
    """
    Standard gradient descent on T².
    Gets stuck at saddle points - no escape mechanism.
    """
    torch.manual_seed(seed)
    theta_A = float(torch.rand(1) * 2 * math.pi)
    theta_B = float(torch.rand(1) * 2 * math.pi)
    tracker = ToroidalPhaseTracker()
    losses = []

    for step in range(steps):
        loss = torus_loss(theta_A, theta_B)
        dA, dB = torus_grad(theta_A, theta_B)
        omega_A = -lr * dA
        omega_B = -lr * dB
        theta_A += omega_A
        theta_B += omega_B
        tracker.step(omega_A, omega_B)
        losses.append(loss)

    wA, wB = tracker.winding_number()
    return losses, tracker.history_A, tracker.history_B, wA, wB


# Langevin Dynamics On Torus
def run_langevin(steps=500, lr=0.05, beta=10.0, seed=42):
    """
    Langevin dynamics on T²:
    θ(t+1) = θ(t) - lr * ∇L + √(2/β) * ξ (Latex for better understanding)

    Noise injection allows escape from mandatory saddle points.
    β controls noise temperature higher β = less noise.
    """
    torch.manual_seed(seed)
    theta_A = float(torch.rand(1) * 2 * math.pi)
    theta_B = float(torch.rand(1) * 2 * math.pi)
    tracker = ToroidalPhaseTracker()
    losses = []
   

    for step in range(steps):
        noise_scale = math.sqrt(2.0 / beta) * (1.0 - step / steps)
        loss = torus_loss(theta_A, theta_B)
        dA, dB = torus_grad(theta_A, theta_B)

        # Langevin - gradient + noise
        noise_A = float(torch.randn(1)) * noise_scale
        noise_B = float(torch.randn(1)) * noise_scale
        omega_A = -lr * dA + noise_A
        omega_B = -lr * dB + noise_B

        theta_A += omega_A
        theta_B += omega_B
        tracker.step(omega_A, omega_B)
        losses.append(loss)

    wA, wB = tracker.winding_number()
    return losses, tracker.history_A, tracker.history_B, wA, wB


# Langevin + Winding Penalty 
def run_langevin_winding(steps=500, lr=0.05, beta=10.0,
                          lambda_w=0.5, seed=42):
    """
    Langevin dynamics + winding penalty on T²:
    L_total = L + λ(w_A² + w_B²) (Latex for better understanding)

    Winding penalty discourages orbiting.
    Forces phase to settle at a fixed point.
    """
    torch.manual_seed(seed)
    theta_A = float(torch.rand(1) * 2 * math.pi)
    theta_B = float(torch.rand(1) * 2 * math.pi)
    tracker = ToroidalPhaseTracker()
    losses = []
  

    for step in range(steps):
        noise_scale = math.sqrt(2.0 / beta) * (1.0 - step / steps)
        loss = torus_loss(theta_A, theta_B)
        dA, dB = torus_grad(theta_A, theta_B)

        # Winding numbers so far
        wA, wB = tracker.winding_number()

        # Winding penalty gradient
        winding_grad_A = 2 * lambda_w * wA
        winding_grad_B = 2 * lambda_w * wB

        # Langevin + winding penalty
        noise_A = float(torch.randn(1)) * noise_scale
        noise_B = float(torch.randn(1)) * noise_scale
        omega_A = -lr * (dA + winding_grad_A) + noise_A
        omega_B = -lr * (dB + winding_grad_B) + noise_B

        theta_A += omega_A
        theta_B += omega_B
        tracker.step(omega_A, omega_B)
        losses.append(loss)

    wA, wB = tracker.winding_number()
    return losses, tracker.history_A, tracker.history_B, wA, wB


# Verification
def verify_toroidal_dynamics():
    """
    Runs all three methods and compares:
    1. Standard GD gets stuck
    2. Langevin escapes saddles
    3. Langevin + winding penalty = escapes + converges
    """
    print("=" * 60)
    print("Toroidal Dynamics Verification")
    print("=" * 60)

    steps = 2000
    results = {}

    print("\n1. Standard Gradient Descent...")
    l1, hA1, hB1, wA1, wB1 = run_standard_gd(steps=steps)
    results['Standard GD'] = (l1, hA1, hB1, wA1, wB1)
    print(f"   Final loss:     {l1[-1]:.6f}")
    print(f"   Winding (A, B): ({wA1:.2f}, {wB1:.2f})")

    print("\n2. Langevin Dynamics...")
    l2, hA2, hB2, wA2, wB2 = run_langevin(steps=steps, beta=50.0)
    results['Langevin'] = (l2, hA2, hB2, wA2, wB2)
    print(f"   Final loss:     {l2[-1]:.6f}")
    print(f"   Winding (A, B): ({wA2:.2f}, {wB2:.2f})")

    print("\n3. Langevin + Winding Penalty...")
    l3, hA3, hB3, wA3, wB3 = run_langevin_winding(steps=steps, beta=50.0, lambda_w=2.0)
    results['Langevin + Winding'] = (l3, hA3, hB3, wA3, wB3)
    print(f"   Final loss:     {l3[-1]:.6f}")
    print(f"   Winding (A, B): ({wA3:.2f}, {wB3:.2f})")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = ['#E8593C', '#3B8BD4', '#1D9E75']
    names = list(results.keys())

    for i, (name, (loss, hA, hB, wA, wB)) in enumerate(results.items()):
        # Loss curve
        axes[0, i].plot(loss, color=colors[i], linewidth=1.5)
        axes[0, i].set_title(f'{name}\nFinal loss: {loss[-1]:.4f}',
                              fontsize=11)
        axes[0, i].set_xlabel('Step')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True, alpha=0.3)

        # Torus path
        axes[1, i].scatter(hA, hB, c=range(len(hA)),
                          cmap='viridis', s=2, alpha=0.6)
        axes[1, i].set_title(f'Phase path on T²\nWinding: '
                              f'({wA:.1f}, {wB:.1f})', fontsize=11)
        axes[1, i].set_xlabel('θ_A')
        axes[1, i].set_ylabel('θ_B')
        axes[1, i].set_xlim(0, 2 * math.pi)
        axes[1, i].set_ylim(0, 2 * math.pi)
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle('Ophanim: Toroidal Training Dynamics',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('toroidal_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved: toroidal_dynamics.png")

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"{'Method':<25} {'Final Loss':>12} {'Winding A':>10} "
          f"{'Winding B':>10}")
    print("-" * 60)
    for name, (loss, _, _, wA, wB) in results.items():
        print(f"{name:<25} {loss[-1]:>12.6f} {wA:>10.2f} {wB:>10.2f}")

    print("\nKey checks:")
    print(f"  Langevin finds same minimum as GD: "
      f"{abs(l2[-1] - l1[-1]) < 0.01}")
    print(f"  Winding penalty reduces orbiting: "
      f"{abs(wA3) + abs(wB3) < abs(wA2) + abs(wB2)}")


if __name__ == "__main__":
    verify_toroidal_dynamics()