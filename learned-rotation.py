"""
Ophanim: Learned Low-Rank Rotation
Replaces fixed (0,1) plane rotation with data-driven
low-rank skew-symmetric rotation.

From 10_Open_Questions.md:
- Current: Ω fixed to (0,1) plane: human assumption
- New: Ω = U V^T - V U^T: learned by gradient descent
- r = O(log d) planes sufficient (Johnson-Lindenstrauss)
- Data decides which planes matter not the human

What stays the same:
- Cayley map: orthogonal matrix guaranteed
- QR reprojection: drift prevention
- Toroidal dynamics: phase state on T²
- Everything else in the architecture

What changes:
- rotate() LearnedLowRankRotation module
- Fixed angles θ_A, θ_B → learned Ω_A, Ω_B
"""

import torch
import torch.nn as nn
import math


# Cayley Map
def cayley(A):
    """
    Cayley map: skew-symmetric A → orthogonal R
    R = (I - A)(I + A)^{-1}
    Guaranteed orthogonal for any skew-symmetric A.
    """
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


#  Learned Low-Rank Rotation
class LearnedLowRankRotation(nn.Module):
    """
    Replaces fixed (0,1) plane rotation.
    
    Learns which rotation planes matter for the task.
    Ω = U V^T - V U^T  (skew-symmetric by construction)
    R = cayley(Ω)       (orthogonal by construction)
    
    r = O(log d) planes sufficient for coverage.
    """
    def __init__(self, d, r=None):
        super().__init__()
        self.d = d
        # r defaults to O(log d): Johnson-Lindenstrauss minimum
        self.r = r if r is not None else max(4, int(math.log2(d)))

        # Learned plane coefficients
        self.alpha = nn.Parameter(torch.randn(self.r) * 0.01)

        # Low-rank factors: define the rotation planes
        self.U = nn.Parameter(torch.randn(d, self.r) * 0.1)
        self.V = nn.Parameter(torch.randn(d, self.r) * 0.1)

    def get_omega(self):
        """
        Build skew-symmetric Ω from learned factors.
        Ω = Σ α_k (u_k v_k^T - v_k u_k^T)
        Skew-symmetric by construction — Ω^T = -Ω always.
        """
        Omega = torch.zeros(self.d, self.d,
                           device=self.U.device)
        for k in range(self.r):
            u_k = self.U[:, k]
            v_k = self.V[:, k]
            # Outer products: skew-symmetric combination
            G_k = torch.outer(u_k, v_k) - torch.outer(v_k, u_k)
            Omega = Omega + self.alpha[k] * G_k
        return Omega

    def forward(self, x):
        """
        x: (batch, d) or (batch, seq, d)
        Returns: rotated x — same shape
        """
        Omega = self.get_omega()
        R = cayley(Omega)           # (d, d) orthogonal
        return x @ R.T

    def active_planes(self, top_k=5):
        """
        Interpretability: which planes have largest weight.
        Returns top_k alpha indices and their values.
        """
        with torch.no_grad():
            vals, idx = self.alpha.abs().topk(top_k)
        return idx.tolist(), vals.tolist()

    def orthogonality_check(self):
        """
        Verify R stays orthogonal.
        Returns ||R^T R - I||  (should be near zero.)
        """
        with torch.no_grad():
            Omega = self.get_omega()
            R = cayley(Omega)
            drift = (R.T @ R - torch.eye(
                self.d, device=R.device)).abs().mean()
        return drift.item()


# Verification 
def verify_learned_rotation():
    print("=" * 55)
    print("Learned Low-Rank Rotation — Verification")
    print("=" * 55)

    torch.manual_seed(42)
    d = 32
    r = max(4, int(math.log2(d)))  # = 5 for d=32

    rot = LearnedLowRankRotation(d=d, r=r)

    print(f"\nDimension d: {d}")
    print(f"Rank r (O(log d)): {r}")
    print(f"Planes covered: {r} of {d*(d-1)//2} possible")

    # Test 1: orthogonality
    drift = rot.orthogonality_check()
    print(f"\nTest 1 — Orthogonality")
    print(f"  ||R^T R - I|| = {drift:.8f} (should be ~0)")

    # Test 2: shape preserved
    x = torch.randn(16, d)
    out = rot(x)
    print(f"\nTest 2 — Shape preserved")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Match:  {x.shape == out.shape}")

    # Test 3: norm preserved (rotation preserves distances)
    norm_in  = x.norm(dim=1).mean().item()
    norm_out = out.norm(dim=1).mean().item()
    print(f"\nTest 3 — Norm preserved (isometry)")
    print(f"  Input norm:  {norm_in:.6f}")
    print(f"  Output norm: {norm_out:.6f}")
    print(f"  Preserved:   {abs(norm_in - norm_out) < 0.001}")

    # Test 4: active planes (interpretability)
    idx, vals = rot.active_planes(top_k=3)
    print(f"\nTest 4 — Active planes (interpretability)")
    print(f"  Top 3 plane indices: {idx}")
    print(f"  Top 3 alpha values:  {[f'{v:.4f}' for v in vals]}")

    # Test 5: gradients flow
    x = torch.randn(16, d)
    out = rot(x)
    loss = out.sum()
    loss.backward()
    grad_U = rot.U.grad is not None
    grad_V = rot.V.grad is not None
    grad_a = rot.alpha.grad is not None
    print(f"\nTest 5 — Gradients flow")
    print(f"  U gradient: {grad_U}")
    print(f"  V gradient: {grad_V}")
    print(f"  α gradient: {grad_a}")

    print(f"\n{'='*55}")
    print("Key checks:")
    print(f"  Orthogonal:      {drift < 1e-5}")
    print(f"  Shape preserved: {x.shape == out.shape}")
    print(f"  Norm preserved:  {abs(norm_in - norm_out) < 0.001}")
    print(f"  Gradients flow:  {grad_U and grad_V and grad_a}")


if __name__ == "__main__":
    verify_learned_rotation()