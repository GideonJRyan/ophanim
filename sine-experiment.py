"""
Shifting Sine Wave Experiment
Tests whether rotating attention tracks frequency shifts
better than static attention.

Signal: sine wave that shifts frequency mid-sequence
Task: predict next value
Models: Baseline, Random 2-Stream, Ophanim
"""

import torch
import torch.nn as nn
import numpy as np
import math


# Data Generation 
def generate_shifting_sine(n_samples=1000, seq_len=100):
    """
    Generate sine waves that shift frequency at the midpoint.
    First half: low frequency
    Second half: high frequency
    Task: predict next value given sequence
    """
    X, y = [], []
    for _ in range(n_samples):
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        # Low frequency first half
        f1 = np.random.uniform(0.5, 1.5)
        # High frequency second half
        f2 = np.random.uniform(3.0, 5.0)

        signal = np.zeros(seq_len + 1)
        mid = seq_len // 2
        signal[:mid] = np.sin(f1 * t[:mid])
        signal[mid:] = np.sin(f2 * t[mid:])

        X.append(signal[:seq_len])
        y.append(signal[1:seq_len + 1])

    X = torch.FloatTensor(np.array(X)).unsqueeze(-1)  # (N, seq_len, 1)
    y = torch.FloatTensor(np.array(y)).unsqueeze(-1)  # (N, seq_len, 1)
    return X, y


# Models 
class SineBaseline(nn.Module):
    """Standard attention on sequence."""
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out   = nn.Linear(d, 1)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.attn(x, x, x)
        return self.out(x)


class SineRandom2Stream(nn.Module):
    """Two streams, no orthogonality guarantee."""
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.embed_A = nn.Linear(1, d//2)
        self.embed_B = nn.Linear(1, d//2)
        self.attn_A  = nn.MultiheadAttention(d//2, heads, batch_first=True)
        self.attn_B  = nn.MultiheadAttention(d//2, heads, batch_first=True)
        self.out     = nn.Linear(d, 1)

    def forward(self, x):
        a = self.embed_A(x)
        b = self.embed_B(x)
        a, _ = self.attn_A(a, a, a)
        b, _ = self.attn_B(b, b, b)
        return self.out(torch.cat([a, b], dim=-1))


class SineOphanim(nn.Module):
    """Ophanim with rotating attention on sequence."""
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.d = d
        self.p = d // 2  # 16

        self.embed  = nn.Linear(1, d)
        self.W_A    = nn.Linear(d, self.p)
        self.W_B    = nn.Linear(d, self.p)
        self.omega_A = nn.Linear(d, 1)
        self.omega_B = nn.Linear(d, 1)
        self.attn_A = nn.MultiheadAttention(self.p, heads, batch_first=True)
        self.attn_B = nn.MultiheadAttention(self.p, heads, batch_first=True)
        self.out    = nn.Linear(d, 1)

        # Orthogonal bases
        U_A, _ = torch.linalg.qr(torch.randn(self.p, self.p))
        U_B, _ = torch.linalg.qr(torch.randn(self.p, self.p))
        self.register_buffer('U_A', U_A)
        self.register_buffer('U_B', U_B)

    def rotate(self, U, theta):
        dim = U.size(0)
        R = torch.eye(dim, device=U.device)
        c = torch.cos(theta).squeeze()
        s = torch.sin(theta).squeeze()
        R[0, 0] = c;  R[0, 1] = -s
        R[1, 0] = s;  R[1, 1] = c
        return R @ U

    def forward(self, x):
        z = self.embed(x)                          # (batch, seq, d)
        theta_A = self.omega_A(z).mean()
        theta_B = self.omega_B(z).mean()

        U_A_rot = self.rotate(self.U_A, theta_A)
        U_B_rot = self.rotate(self.U_B, theta_B)

        a = self.W_A(z) @ U_A_rot.T               # (batch, seq, p)
        b = self.W_B(z) @ U_B_rot.T               # (batch, seq, p)

        h_A, _ = self.attn_A(a, a, a)
        h_B, _ = self.attn_B(b, b, b)

        return self.out(torch.cat([h_A, h_B], dim=-1))


# Training
def train_sine(model, X_train, y_train, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>2} loss: {loss.item():.6f}")
    return loss.item()


# Evaluation
def evaluate_sine(model, X_test, y_test):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        pred = model(X_test)
        loss = criterion(pred, y_test).item()

        # Split evaluation - first half vs second half
        mid = y_test.size(1) // 2
        loss_first  = criterion(pred[:, :mid],  y_test[:, :mid]).item()
        loss_second = criterion(pred[:, mid:],  y_test[:, mid:]).item()

    return loss, loss_first, loss_second


# Main 
if __name__ == "__main__":
    torch.manual_seed(42)

    # Generate data
    X, y = generate_shifting_sine(n_samples=1000, seq_len=100)
    split = 800
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    models = {
        "Baseline":        SineBaseline(),
        "Random 2-Stream": SineRandom2Stream(),
        "Ophanim":         SineOphanim(),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"Training {name}...")
        print(f"{'-'*50}")
        train_sine(model, X_train, y_train, epochs=100)
        loss, l1, l2 = evaluate_sine(model, X_test, y_test)
        results[name] = (loss, l1, l2)
        print(f"  Total MSE:       {loss:.6f}")
        print(f"  First half MSE:  {l1:.6f}  (low freq)")
        print(f"  Second half MSE: {l2:.6f}  (high freq — shift)")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'Total':>10} {'Low Freq':>10} {'High Freq':>10}")
    print("-" * 52)
    for name, (total, l1, l2) in results.items():
        print(f"{name:<20} {total:>10.6f} {l1:>10.6f} {l2:>10.6f}")

    print("\nLower MSE = better prediction.")
    print("Key metric: High Freq MSE - tests frequency shift tracking.")