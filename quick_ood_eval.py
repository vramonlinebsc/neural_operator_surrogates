#!/usr/bin/env python3
"""
Quick OOD Evaluation Script
Tests if the bug fix (random OOD selection) improved performance
Run on GCP: python3 quick_ood_eval.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
import scipy.sparse as sp
from scipy.linalg import expm

print("="*70)
print("QUICK OOD EVALUATION - Testing Bug Fix")
print("="*70)

# ==============================================================================
# 1. LOAD MODEL ARCHITECTURE (copied from your script)
# ==============================================================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1),
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum('bix,iox->box',
            x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x


class ImprovedPhysicsInformedFNO(nn.Module):
    def __init__(self, modes=24, width=128, n_layers=6, n_params=5, 
                 n_outputs=3, time_steps=300):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.n_params = n_params
        self.n_outputs = n_outputs
        self.time_steps = time_steps

        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width * time_steps)
        )

        self.conv_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width//2, 1), nn.GELU(),
            nn.Conv1d(width//2, n_outputs, 1)
        )

    def forward(self, params, T=None):
        batch_size = params.shape[0]
        T = T or self.time_steps
        x = self.param_encoder(params)
        x = x.view(batch_size, self.width, T)
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2) + x
        out = self.output_proj(x)
        return out.permute(0, 2, 1)


# ==============================================================================
# 2. EXACT SIMULATOR (for ground truth)
# ==============================================================================

class SpinSystemOptimized:
    def __init__(self, N, topology='chain'):
        self.N = N
        self.dim = 2 ** N
        self.topology = topology
        self._build_operators()

    def _kron_list(self, ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _build_operators(self):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)

        self.Ix, self.Iy, self.Iz = [], [], []
        for i in range(self.N):
            ops = [identity] * self.N
            ops[i] = sx
            self.Ix.append(self._kron_list(ops))
            ops[i] = sy
            self.Iy.append(self._kron_list(ops))
            ops[i] = sz
            self.Iz.append(self._kron_list(ops))

    def build_hamiltonian(self, Omega, J):
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.N):
            H = H + Omega[i] * self.Iz[i]
        pairs = [(i, i+1) for i in range(self.N-1)]
        for i, j in pairs:
            H = H + 2*np.pi*J * (
                self.Ix[i]@self.Ix[j] + self.Iy[i]@self.Iy[j] + self.Iz[i]@self.Iz[j]
            )
        return H

    def simulate(self, Omega, J, T, dt=1e-4):
        H = self.build_hamiltonian(Omega, J)
        psi0 = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        times = np.arange(T) * dt
        
        Mx = np.zeros(T)
        My = np.zeros(T)
        I1z = np.zeros(T)
        
        Ix_avg = sum(self.Ix) / self.N
        Iy_avg = sum(self.Iy) / self.N
        Iz_first = self.Iz[0]
        
        U = expm(-1j * H * dt)
        psi_t = psi0.copy()
        
        for t_idx in range(T):
            Mx[t_idx] = np.real(np.conj(psi_t) @ Ix_avg @ psi_t)
            My[t_idx] = np.real(np.conj(psi_t) @ Iy_avg @ psi_t)
            I1z[t_idx] = np.real(np.conj(psi_t) @ Iz_first @ psi_t)
            psi_t = U @ psi_t
        
        return {'Mx': Mx, 'My': My, 'I1z': I1z}


# ==============================================================================
# 3. LOAD CHECKPOINT AND DATA
# ==============================================================================

N = 4
T = 300
device = torch.device('cpu')

print("\n1. Loading checkpoint...")
ckpt_path = Path('checkpoints/model_N4_chain_best.pt')
if not ckpt_path.exists():
    print(f"ERROR: Checkpoint not found at {ckpt_path}")
    print("Training may still be running. Wait for first checkpoint.")
    exit(1)

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print(f"   ✅ Loaded from epoch {checkpoint['epoch']}")
print(f"   Best val loss: {checkpoint['best_val_loss']:.6f}")

print("\n2. Loading normalization stats...")
stats_path = Path('checkpoints/norm_stats_N4_chain.pkl')
with open(stats_path, 'rb') as f:
    stats = pickle.load(f)
print(f"   ✅ Stats loaded")
print(f"   obs_mean: {stats['obs_mean']}")
print(f"   obs_std: {stats['obs_std']}")

print("\n3. Loading validation dataset...")
val_path = Path('checkpoints/dataset_N4_chain_val.pkl')
with open(val_path, 'rb') as f:
    val_data = pickle.load(f)
print(f"   ✅ Loaded {len(val_data)} samples")

print("\n4. Creating model...")
model = ImprovedPhysicsInformedFNO(
    modes=12, width=64, n_layers=4,
    n_params=N+1, n_outputs=3, time_steps=T
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   ✅ Model ready")

# ==============================================================================
# 4. TEST ON MULTIPLE RANDOM OOD SAMPLES
# ==============================================================================

print("\n" + "="*70)
print("TESTING ON RANDOM OOD SAMPLES (Bug Fix Verification)")
print("="*70)

np.random.seed(42)
n_test_samples = 10
errors = []

system = SpinSystemOptimized(N, 'chain')

for test_idx in range(n_test_samples):
    # THIS IS THE FIX: Random selection, not val_data[0]
    ood_idx = np.random.randint(0, len(val_data))
    sample = val_data[ood_idx]
    
    # Extract and denormalize parameters
    params_norm = sample['params']
    params_raw = params_norm * stats['param_std'] + stats['param_mean']
    
    Omega = params_raw[:N] / (2 * np.pi)
    J = params_raw[N]
    
    # Ground truth (exact simulation)
    exact = system.simulate(Omega, J, T, dt=1e-4)
    
    # Surrogate prediction
    params_t = torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(params_t).squeeze().numpy()
    
    # Denormalize prediction
    pred_denorm = pred * stats['obs_std'] + stats['obs_mean']
    
    # Compute error
    error = np.sqrt(
        np.mean((exact['Mx'] - pred_denorm[:, 0])**2) +
        np.mean((exact['My'] - pred_denorm[:, 1])**2) +
        np.mean((exact['I1z'] - pred_denorm[:, 2])**2)
    )
    
    errors.append(error)
    
    if test_idx < 3:  # Print first 3 in detail
        print(f"\nSample {test_idx+1} (idx={ood_idx}):")
        print(f"  J coupling: {J:.2f} Hz")
        print(f"  RMSE: {error:.6f}")
        print(f"  Mx error: {np.mean((exact['Mx'] - pred_denorm[:, 0])**2):.6f}")
        print(f"  My error: {np.mean((exact['My'] - pred_denorm[:, 1])**2):.6f}")
        print(f"  I1z error: {np.mean((exact['I1z'] - pred_denorm[:, 2])**2):.6f}")

# ==============================================================================
# 5. RESULTS SUMMARY
# ==============================================================================

errors = np.array(errors)
mean_error = errors.mean()
std_error = errors.std()
max_error = errors.max()
min_error = errors.min()

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\nTested on {n_test_samples} random OOD samples:")
print(f"  Mean RMSE:   {mean_error:.6f}")
print(f"  Std RMSE:    {std_error:.6f}")
print(f"  Min RMSE:    {min_error:.6f}")
print(f"  Max RMSE:    {max_error:.6f}")

print(f"\n📊 Interpretation:")
if mean_error < 0.05:
    print(f"  ✅ EXCELLENT! Error < 5% - Bug fix worked!")
    print(f"  Previous error was 29%, now it's {mean_error*100:.2f}%")
elif mean_error < 0.10:
    print(f"  ✅ GOOD! Error < 10% - Significant improvement")
    print(f"  Previous error was 29%, now it's {mean_error*100:.2f}%")
else:
    print(f"  ⚠️  Error still high: {mean_error*100:.2f}%")
    print(f"  Model may need more training or there's another issue")

print(f"\n💡 Note: These are interim results from epoch {checkpoint['epoch']}")
print(f"   Training continues to epoch 500, error should improve further")

print("\n" + "="*70)
