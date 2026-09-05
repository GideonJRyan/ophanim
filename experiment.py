import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Data 
transform = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.ToTensor()
])

train = datasets.MNIST('./data', train=True, download=False, transform=transform)
test  = datasets.MNIST('./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=64, shuffle=False)

# Baseline: Standard Attention
class Baseline(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.embed   = nn.Linear(28*28, d)
        self.attn    = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out     = nn.Linear(d, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)       # flatten
        x = self.embed(x).unsqueeze(1)   # embed
        x, _ = self.attn(x, x, x)        # attend
        x = x.squeeze(1)
        return self.out(x)

# Ablation 1: Random Two-Stream (no orthogonality) 
class RandomTwoStream(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.embed_A = nn.Linear(28*28, d//2)
        self.embed_B = nn.Linear(28*28, d//2)
        self.attn_A  = nn.MultiheadAttention(d//2, heads, batch_first=True)
        self.attn_B  = nn.MultiheadAttention(d//2, heads, batch_first=True)
        self.out     = nn.Linear(d, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a = self.embed_A(x).unsqueeze(1)
        b = self.embed_B(x).unsqueeze(1)
        a, _ = self.attn_A(a, a, a)
        b, _ = self.attn_B(b, b, b)
        merged = torch.cat([a.squeeze(1), b.squeeze(1)], dim=1)
        return self.out(merged)

# Ablation 2: Ophanim 
class Ophanim(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.d = d
        self.p = d // 2  # 32
        self.q = d // 2  # 32

        # Spirit vector encoder: reads full input
        self.W_z = nn.Linear(28*28, d)

        # Readout into each subspace
        self.W_A = nn.Linear(d, self.p)
        self.W_B = nn.Linear(d, self.q)

        # Learned rotation rates from spirit
        self.omega_A = nn.Linear(d, 1)
        self.omega_B = nn.Linear(d, 1)

        # Attention within each stream: input dim = p = 32
        self.attn_A = nn.MultiheadAttention(self.p, heads, batch_first=True)
        self.attn_B = nn.MultiheadAttention(self.q, heads, batch_first=True)

        # Output: takes p + q = 64
        self.out = nn.Linear(d, 10)

        # Orthogonal subspace bases: p×p and q×q
        U_A, _ = torch.linalg.qr(torch.randn(self.p, self.p))
        U_B, _ = torch.linalg.qr(torch.randn(self.q, self.q))
        self.register_buffer('U_A', U_A)
        self.register_buffer('U_B', U_B)

    def rotate(self, U, theta):
        """Rotate matrix U by angle theta in the (0,1) plane."""
        dim = U.size(0)
        R = torch.eye(dim, device=U.device)
        c = torch.cos(theta).squeeze()
        s = torch.sin(theta).squeeze()
        R[0, 0] = c;  R[0, 1] = -s
        R[1, 0] = s;  R[1, 1] = c
        return R @ U

    def forward(self, x):
        batch = x.size(0)
        x_flat = x.view(batch, -1)          # (batch, 784)

        # Spirit vector reads everything
        z = self.W_z(x_flat)                # (batch, 64)

        # Rotation rates from spirit
        theta_A = self.omega_A(z).mean()    # scalar
        theta_B = self.omega_B(z).mean()    # scalar

        # Rotate subspace bases
        U_A_rot = self.rotate(self.U_A, theta_A)   # (32, 32)
        U_B_rot = self.rotate(self.U_B, theta_B)   # (32, 32)

        # Project spirit into each subspace
        alpha_A = self.W_A(z)               # (batch, 32)
        alpha_B = self.W_B(z)               # (batch, 32)

        # Apply rotation
        v_A = alpha_A @ U_A_rot.T           # (batch, 32)
        v_B = alpha_B @ U_B_rot.T           # (batch, 32)

        # Attend within each stream
        h_A, _ = self.attn_A(
            v_A.unsqueeze(1),
            v_A.unsqueeze(1),
            v_A.unsqueeze(1)
        )                                   # (batch, 1, 32)

        h_B, _ = self.attn_B(
            v_B.unsqueeze(1),
            v_B.unsqueeze(1),
            v_B.unsqueeze(1)
        )                                   # (batch, 1, 32)

        # Merge streams
        merged = torch.cat([
            h_A.squeeze(1),
            h_B.squeeze(1)
        ], dim=1)                           # (batch, 64)

        return self.out(merged)             # (batch, 10)

# Training Loop 
def train_model(model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

# Evaluation 
def evaluate(model, angles=[0, 90, 180, 270]):
    model.eval()
    results = {}
    for angle in angles:
        tf = transforms.Compose([
            transforms.RandomRotation((angle, angle)),
            transforms.ToTensor()
        ])
        ds = datasets.MNIST('./data', train=False, download=False, transform=tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=64)
        correct = total = 0
        with torch.no_grad():
            for images, labels in dl:
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        results[angle] = correct / total * 100
    return results

# Run Baseline 
if __name__ == "__main__":
    print("Training Baseline...")
    print("-" * 50)
    baseline = Baseline()
    train_model(baseline)
    results = evaluate(baseline)
    print("\nBaseline Rotation Robustness:")
    for angle, acc in results.items():
        print(f"  {angle:>3}°: {acc:.2f}%")

    print("\n" + "-" * 50)
    print("Training Ablation 1: Random Two-Stream...")
    print("-" * 50)
    ablation1 = RandomTwoStream()
    train_model(ablation1)
    results_a1 = evaluate(ablation1)
    print("\nAblation 1 Rotation Robustness:")
    for angle, acc in results_a1.items():
        print(f"  {angle:>3}°: {acc:.2f}%")

    print("\n" + "=" * 50)
    print("Training Ablation 2: Ophanim...")
    print("=" * 50)
    ophanim = Ophanim()
    train_model(ophanim)
    results_op = evaluate(ophanim)
    print("\nOphanim Rotation Robustness:")
    for angle, acc in results_op.items():
        print(f"  {angle:>3}°: {acc:.2f}%")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Model':<20} {'0°':>6} {'90°':>6} {'180°':>6} {'270°':>6} {'Avg':>6}")
    print("-" * 50)
    for name, res in [("Baseline", results), ("Random 2-Stream", results_a1), ("Ophanim", results_op)]:
        avg = sum(res.values()) / len(res)
        print(f"{name:<20} {res[0]:>5.1f}% {res[90]:>5.1f}% {res[180]:>5.1f}% {res[270]:>5.1f}% {avg:>5.1f}%")
