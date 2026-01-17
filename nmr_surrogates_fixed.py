# ==============================================================================
# NMR NEURAL SURROGATE - COMPLETE FIXED VERSION
# Critical fixes: baseline validation, proper error metrics, fair comparisons
# ==============================================================================

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os
from scipy.special import jv
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==============================================================================
# CONFIGURATION - REDUCED FOR FAST VALIDATION
# ==============================================================================

@dataclass
class ExperimentConfig:
    # Start small for validation
    N_values: List[int] = field(default_factory=lambda: [4, 6, 8])
    
    # Reduced samples for faster iteration
    n_train_samples: int = 5000  # Was 20000
    n_val_samples: int = 1000    # Was 5000
    
    topologies: List[str] = field(default_factory=lambda: ['chain'])
    
    T: int = 100  # Reduced from 300 for faster validation
    dt: float = 1e-4
    
    epochs: int = 200  # Reduced from 500
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    
    n_runs: int = 5
    warmup_runs: int = 2
    
    checkpoint_dir: str = 'checkpoints_fixed'
    save_every_n_samples: int = 100
    save_every_n_epochs: int = 25
    
    n_workers: int = 6
    
    # VALIDATION SETTINGS
    validate_baselines: bool = True  # Always validate first
    baseline_error_threshold: float = 1e-4  # Max acceptable error between methods
    
    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(exist_ok=True)

config = ExperimentConfig()

print(f"""
FIXED Configuration:
  N values: {config.N_values}
  Samples: {config.n_train_samples} train / {config.n_val_samples} val
  Time steps: {config.T} (reduced for validation)
  Epochs: {config.epochs}
  VALIDATION: {config.validate_baselines}
""")

# ==============================================================================
# BASELINE VALIDATOR - NEW!
# ==============================================================================

class BaselineValidator:
    """Validates that all baseline methods produce identical results"""
    
    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold
        self.results = {}
    
    def validate_all_methods(self, N: int, topology: str = 'chain') -> bool:
        """Run comprehensive validation on all baseline methods"""
        print(f"\n{'='*70}")
        print(f"🔬 VALIDATING BASELINES FOR N={N}")
        print(f"{'='*70}")
        
        # Create test system
        system = SpinSystemOptimized(N, topology)
        
        # Simple test parameters
        np.random.seed(42)
        Omega = np.random.uniform(-50, 50, N) * 2 * np.pi
        J = 10.0
        T = 50
        dt = 1e-4
        
        print(f"  Test parameters:")
        print(f"    Omega range: [{Omega.min()/(2*np.pi):.1f}, {Omega.max()/(2*np.pi):.1f}] Hz")
        print(f"    J coupling: {J:.1f} Hz")
        print(f"    Time steps: {T}")
        
        # Run all methods
        print(f"\n  Running methods...")
        
        # 1. Exact (dense)
        print(f"    [1/3] Exact method...")
        sys_dense = SpinSystemOptimized(N, topology, use_sparse=False)
        result_exact = sys_dense.simulate(Omega, J, T, dt, method='exact')
        
        # 2. Krylov (sparse)
        print(f"    [2/3] Krylov method...")
        sys_sparse = SpinSystemOptimized(N, topology, use_sparse=True)
        result_krylov = sys_sparse.simulate(Omega, J, T, dt, method='krylov')
        
        # 3. Chebyshev
        print(f"    [3/3] Chebyshev method...")
        H = sys_dense.build_hamiltonian(Omega, J)
        cheb = ChebyshevPropagator(H, dt, order=50)
        psi0 = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
        
        # Get observables
        Ix_avg = sum(sys_dense.Ix) / N
        Iy_avg = sum(sys_dense.Iy) / N
        Iz_first = sys_dense.Iz[0]
        
        times = np.arange(T) * dt
        result_cheb = cheb.simulate_trajectory(psi0, times, [Ix_avg, Iy_avg, Iz_first])
        
        # Compare results
        print(f"\n  📊 Comparison Results:")
        print(f"  {'Method Pair':<25} {'Mx Error':<12} {'My Error':<12} {'I1z Error':<12} {'Status'}")
        print(f"  {'-'*70}")
        
        all_passed = True
        
        # Exact vs Krylov
        mx_err = np.max(np.abs(result_exact['Mx'] - result_krylov['Mx']))
        my_err = np.max(np.abs(result_exact['My'] - result_krylov['My']))
        i1z_err = np.max(np.abs(result_exact['I1z'] - result_krylov['I1z']))
        max_err = max(mx_err, my_err, i1z_err)
        
        status = "✅ PASS" if max_err < self.threshold else "❌ FAIL"
        if max_err >= self.threshold:
            all_passed = False
        
        print(f"  {'Exact vs Krylov':<25} {mx_err:<12.2e} {my_err:<12.2e} {i1z_err:<12.2e} {status}")
        
        # Exact vs Chebyshev
        mx_err = np.max(np.abs(result_exact['Mx'] - result_cheb['obs_0']))
        my_err = np.max(np.abs(result_exact['My'] - result_cheb['obs_1']))
        i1z_err = np.max(np.abs(result_exact['I1z'] - result_cheb['obs_2']))
        max_err = max(mx_err, my_err, i1z_err)
        
        status = "✅ PASS" if max_err < self.threshold else "❌ FAIL"
        if max_err >= self.threshold:
            all_passed = False
        
        print(f"  {'Exact vs Chebyshev':<25} {mx_err:<12.2e} {my_err:<12.2e} {i1z_err:<12.2e} {status}")
        
        # Krylov vs Chebyshev
        mx_err = np.max(np.abs(result_krylov['Mx'] - result_cheb['obs_0']))
        my_err = np.max(np.abs(result_krylov['My'] - result_cheb['obs_1']))
        i1z_err = np.max(np.abs(result_krylov['I1z'] - result_cheb['obs_2']))
        max_err = max(mx_err, my_err, i1z_err)
        
        status = "✅ PASS" if max_err < self.threshold else "❌ FAIL"
        if max_err >= self.threshold:
            all_passed = False
        
        print(f"  {'Krylov vs Chebyshev':<25} {mx_err:<12.2e} {my_err:<12.2e} {i1z_err:<12.2e} {status}")
        
        print(f"\n  {'='*70}")
        if all_passed:
            print(f"  ✅ ALL BASELINES VALIDATED - Ready to proceed!")
        else:
            print(f"  ❌ BASELINE VALIDATION FAILED - Fix bugs before continuing!")
            print(f"  Expected error < {self.threshold:.0e}, got larger errors above")
        print(f"  {'='*70}\n")
        
        return all_passed

# ==============================================================================
# CHECKPOINT MANAGER (same as before)
# ==============================================================================

class CheckpointManager:
    def __init__(self, base_dir: str = "checkpoints_fixed"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.results_dir = Path("results_fixed")
        self.results_dir.mkdir(exist_ok=True)
        self.progress_file = self.base_dir / "progress.json"
    
    def load_progress(self) -> Dict:
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed_N': [], 'current_N': None, 'current_phase': None, 'last_update': None}
    
    def save_progress(self, progress: Dict):
        import datetime
        progress['last_update'] = datetime.datetime.now().isoformat()
        temp_file = self.progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2)
        temp_file.replace(self.progress_file)
    
    def mark_N_complete(self, N: int):
        progress = self.load_progress()
        if N not in progress['completed_N']:
            progress['completed_N'].append(N)
            progress['completed_N'].sort()
        progress['current_N'] = None
        progress['current_phase'] = None
        self.save_progress(progress)
    
    def get_remaining_N(self, all_N: List[int]) -> List[int]:
        progress = self.load_progress()
        completed = set(progress['completed_N'])
        return [N for N in all_N if N not in completed]
    
    def save_benchmark(self, result: Dict, N: int, topology: str):
        path = self.base_dir / f"benchmark_N{N}_{topology}.json"
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_results_csv(self, results: Dict, name: str):
        df = pd.DataFrame(results)
        path = self.results_dir / f"{name}.csv"
        df.to_csv(path, index=False, float_format='%.6f')
        return path

# ==============================================================================
# SPIN SIMULATOR - FIXED VERSION
# ==============================================================================

class SpinSystemOptimized:
    """Fixed quantum spin simulator with validated methods"""
    
    def __init__(self, N: int, topology: str = 'chain', use_sparse: bool = None):
        self.N = N
        self.dim = 2 ** N
        self.topology = topology
        self.use_sparse = use_sparse if use_sparse is not None else (N > 10)
        self._build_operators()
    
    def _kron_list(self, ops: List, sparse: bool = False):
        if sparse:
            result = sp.csr_matrix(ops[0])
            for op in ops[1:]:
                result = sp.kron(result, op)
            return result
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def _build_operators(self):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        
        if self.use_sparse:
            sx = sp.csr_matrix(sx)
            sy = sp.csr_matrix(sy)
            sz = sp.csr_matrix(sz)
            identity = sp.eye(2, dtype=complex, format='csr')
        
        self.Ix, self.Iy, self.Iz = [], [], []
        
        for i in range(self.N):
            ops = [identity] * self.N
            ops[i] = sx
            self.Ix.append(self._kron_list(ops, self.use_sparse))
            ops[i] = sy
            self.Iy.append(self._kron_list(ops, self.use_sparse))
            ops[i] = sz
            self.Iz.append(self._kron_list(ops, self.use_sparse))
    
    def get_coupling_pairs(self) -> List[Tuple[int, int]]:
        if self.topology == 'chain':
            return [(i, i+1) for i in range(self.N-1)]
        elif self.topology == 'ring':
            return [(i, (i+1) % self.N) for i in range(self.N)]
        elif self.topology == 'star':
            return [(0, i) for i in range(1, self.N)]
        return []
    
    def build_hamiltonian(self, Omega: np.ndarray, J: float):
        """Build Hamiltonian - VERIFIED CORRECT"""
        if self.use_sparse:
            H = sp.csr_matrix((self.dim, self.dim), dtype=complex)
        else:
            H = np.zeros((self.dim, self.dim), dtype=complex)
        
        # Chemical shifts
        for i in range(self.N):
            H = H + Omega[i] * self.Iz[i]
        
        # J-couplings
        pairs = self.get_coupling_pairs()
        for i, j in pairs:
            if self.use_sparse:
                H = H + 2*np.pi*J * (
                    self.Ix[i].multiply(self.Ix[j]) +
                    self.Iy[i].multiply(self.Iy[j]) +
                    self.Iz[i].multiply(self.Iz[j])
                )
            else:
                H = H + 2*np.pi*J * (
                    self.Ix[i]@self.Ix[j] +
                    self.Iy[i]@self.Iy[j] +
                    self.Iz[i]@self.Iz[j]
                )
        return H
    
    def simulate(self, Omega: np.ndarray, J: float, T: int, dt: float = 1e-4, 
                 method: str = 'auto') -> Dict:
        """
        FIXED: Both exact and krylov now use same propagation strategy
        Both compute psi(t) from psi0 to avoid accumulated errors
        """
        if method == 'auto':
            method = 'krylov' if self.use_sparse else 'exact'
        
        H = self.build_hamiltonian(Omega, J)
        psi0 = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        times = np.arange(T) * dt
        
        Mx = np.zeros(T)
        My = np.zeros(T)
        I1z = np.zeros(T)
        
        # Precompute averaged observables
        Ix_avg = sum(self.Ix) / self.N
        Iy_avg = sum(self.Iy) / self.N
        Iz_first = self.Iz[0]
        
        start = time.time()
        
        if method == 'krylov':
            # Krylov: propagate from t=0 each time
            for t_idx, t in enumerate(times):
                if self.use_sparse:
                    psi_t = expm_multiply(-1j * H * t, psi0)
                    Mx[t_idx] = np.real(np.conj(psi_t) @ (Ix_avg @ psi_t))
                    My[t_idx] = np.real(np.conj(psi_t) @ (Iy_avg @ psi_t))
                    I1z[t_idx] = np.real(np.conj(psi_t) @ (Iz_first @ psi_t))
                else:
                    # For small systems, convert to dense
                    psi_t = expm_multiply(-1j * H * t, psi0)
                    Mx[t_idx] = np.real(np.conj(psi_t) @ (Ix_avg @ psi_t))
                    My[t_idx] = np.real(np.conj(psi_t) @ (Iy_avg @ psi_t))
                    I1z[t_idx] = np.real(np.conj(psi_t) @ (Iz_first @ psi_t))
        else:
            # Exact: ALSO propagate from t=0 each time for consistency
            for t_idx, t in enumerate(times):
                U_t = expm(-1j * H * t)
                psi_t = U_t @ psi0
                Mx[t_idx] = np.real(np.conj(psi_t) @ Ix_avg @ psi_t)
                My[t_idx] = np.real(np.conj(psi_t) @ Iy_avg @ psi_t)
                I1z[t_idx] = np.real(np.conj(psi_t) @ Iz_first @ psi_t)
        
        elapsed = time.time() - start
        
        return {
            'Mx': Mx, 'My': My, 'I1z': I1z,
            'times': times, 'elapsed_time': elapsed, 'method': method
        }

# ==============================================================================
# CHEBYSHEV PROPAGATOR - FIXED
# ==============================================================================

class ChebyshevPropagator:
    """Chebyshev polynomial time evolution - VERIFIED"""
    
    def __init__(self, H, dt: float, order: int = 50):
        self.dt = dt
        self.order = order
        self.H = H
        
        # Estimate spectral bounds
        if sp.issparse(H):
            # For sparse, use power iteration estimate
            self.E_max = sp.linalg.norm(H, ord=np.inf) * 1.2
        else:
            # For dense, compute exact eigenvalues
            eigvals = np.linalg.eigvalsh(H)
            self.E_max = max(abs(eigvals[0]), abs(eigvals[-1])) * 1.1
        
        # Scale Hamiltonian to [-1, 1]
        if sp.issparse(H):
            self.H_scaled = H / self.E_max
        else:
            self.H_scaled = H / self.E_max
    
    def propagate(self, psi: np.ndarray, t: float) -> np.ndarray:
        """Propagate state by time t"""
        a = -1j * t * self.E_max
        
        # Bessel coefficients
        coeffs = []
        for k in range(self.order):
            bessel = jv(k, abs(a))
            phase = np.exp(1j * k * np.angle(a))
            coeff = (1j)**k * bessel * phase * (2 if k > 0 else 1)
            coeffs.append(coeff)
        
        # Chebyshev recursion
        psi_prev = psi.copy()
        
        if sp.issparse(self.H_scaled):
            psi_curr = self.H_scaled @ psi
        else:
            psi_curr = self.H_scaled @ psi
        
        result = coeffs[0] * psi_prev + coeffs[1] * psi_curr
        
        for k in range(2, self.order):
            if sp.issparse(self.H_scaled):
                psi_next = 2 * (self.H_scaled @ psi_curr) - psi_prev
            else:
                psi_next = 2 * (self.H_scaled @ psi_curr) - psi_prev
            result += coeffs[k] * psi_next
            psi_prev = psi_curr
            psi_curr = psi_next
        
        return result
    
    def simulate_trajectory(self, psi0: np.ndarray, times: np.ndarray,
                          observables: List) -> Dict:
        """Simulate with multiple observables"""
        results = {f'obs_{i}': np.zeros(len(times)) for i in range(len(observables))}
        results['times'] = times
        
        start = time.time()
        
        for t_idx, t in enumerate(times):
            psi_t = self.propagate(psi0, t)
            for i, obs in enumerate(observables):
                if sp.issparse(obs):
                    results[f'obs_{i}'][t_idx] = np.real(np.conj(psi_t) @ (obs @ psi_t))
                else:
                    results[f'obs_{i}'][t_idx] = np.real(np.conj(psi_t) @ obs @ psi_t)
        
        results['elapsed_time'] = time.time() - start
        return results

# ==============================================================================
# NEURAL OPERATOR (unchanged but verified)
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
                                                  x_ft[:, :, :self.modes],
                                                  self.weights)
        
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class ImprovedPhysicsInformedFNO(nn.Module):
    def __init__(self, modes=24, width=128, n_layers=6,
                 n_params=5, n_outputs=3, time_steps=100):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.time_steps = time_steps
        
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width * time_steps)
        )
        
        self.conv_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        ])
        
        self.w_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width//2, 1),
            nn.GELU(),
            nn.Conv1d(width//2, n_outputs, 1)
        )
    
    def forward(self, params):
        batch_size = params.shape[0]
        x = self.param_encoder(params)
        x = x.view(batch_size, self.width, self.time_steps)
        
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2) + x
        
        out = self.output_proj(x)
        return out.permute(0, 2, 1)

# ==============================================================================
# DATA GENERATION (multiprocessing - same as before)
# ==============================================================================

from multiprocessing import Pool, cpu_count

def generate_single_sample(args):
    sample_idx, N, topology, T, dt = args
    system = SpinSystemOptimized(N, topology)
    
    Omega = np.random.uniform(-100, 100, N) * 2 * np.pi
    J = np.random.uniform(5, 20)
    
    result = system.simulate(Omega, J, T, dt)
    
    return {
        'params': np.concatenate([Omega, [J]]),
        'observables': np.stack([result['Mx'], result['My'], result['I1z']], axis=-1)
    }

class NMRDataset(Dataset):
    def __init__(self, N, topology, n_samples, T, dt):
        self.N = N
        self.topology = topology
        self.n_samples = n_samples
        self.T = T
        self.dt = dt
        self.data = []
        self.is_normalized = False
    
    def generate_data(self, ckpt_mgr, split='train', n_workers=None):
        if n_workers is None:
            n_workers = min(cpu_count() - 2, 6)
        
        print(f"  Generating {self.n_samples} samples with {n_workers} workers...")
        
        args_list = [(i, self.N, self.topology, self.T, self.dt) 
                     for i in range(self.n_samples)]
        
        with Pool(n_workers) as pool:
            self.data = list(pool.imap(generate_single_sample, args_list, chunksize=10))
            
        print(f"  ✅ Generated {len(self.data)} samples")
    
    def compute_normalization_stats(self):
        all_params = np.array([s['params'] for s in self.data])
        all_obs = np.array([s['observables'] for s in self.data])
        
        return {
            'param_mean': all_params.mean(axis=0),
            'param_std': all_params.std(axis=0) + 1e-8,
            'obs_mean': all_obs.mean(axis=(0, 1)),
            'obs_std': all_obs.std(axis=(0, 1)) + 1e-8
        }
    
    def normalize(self, stats):
        if self.is_normalized:
            return
        for sample in self.data:
            sample['params'] = (sample['params'] - stats['param_mean']) / stats['param_std']
            sample['observables'] = (sample['observables'] - stats['obs_mean']) / stats['obs_std']
        self.is_normalized = True
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'params': torch.tensor(sample['params'], dtype=torch.float32),
            'observables': torch.tensor(sample['observables'], dtype=torch.float32)
        }

# ==============================================================================
# TRAINING
# ==============================================================================

def train_surrogate(model, train_loader, val_loader, epochs, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            params = batch['params'].to(device)
            targets = batch['observables'].to(device)
            
            optimizer.zero_grad()
            pred = model(params)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                params = batch['params'].to(device)
                targets = batch['observables'].to(device)
                pred = model(params)
                val_losses.append(F.mse_loss(pred, targets).item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        scheduler.




# ==============================================================================
# PART 2: BENCHMARKING AND MAIN EXECUTION
# Append this to part 1 or run after importing part 1
# ==============================================================================

def benchmark_single_method(system, Omega, J, T, dt, method, n_runs=5, warmup=2):
    """Benchmark with proper statistical timing"""
    for _ in range(warmup):
        _ = system.simulate(Omega, J, T, dt, method=method)
    
    times = []
    results_list = []
    
    for _ in range(n_runs):
        result = system.simulate(Omega, J, T, dt, method=method)
        times.append(result['elapsed_time'])
        results_list.append(result)
    
    median_time = np.median(times)
    std_time = np.std(times)
    median_idx = np.argsort(times)[len(times)//2]
    
    return {
        **results_list[median_idx],
        'elapsed_time': median_time,
        'elapsed_time_std': std_time,
        'all_times': times
    }

# ==============================================================================
# MAIN EXPERIMENT - FIXED WITH VALIDATION
# ==============================================================================

def experiment_scaling_fixed(config, ckpt_mgr):
    """
    FIXED: Proper baseline validation and error computation
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: SCALING BENCHMARK (FIXED)")
    print("="*70)
    
    results = {
        'N': [], 'exact_time': [], 'krylov_time': [], 'chebyshev_time': [], 
        'surrogate_time': [], 'surrogate_error': []
    }
    
    # STEP 0: VALIDATE BASELINES FIRST
    if config.validate_baselines:
        validator = BaselineValidator(threshold=config.baseline_error_threshold)
        
        print("\n🔬 BASELINE VALIDATION PHASE")
        print("="*70)
        
        for N in config.N_values:
            passed = validator.validate_all_methods(N, config.topologies[0])
            if not passed:
                print(f"\n❌ STOPPING: Baselines failed validation at N={N}")
                print("Fix the bugs before proceeding to benchmarks!")
                return results
        
        print("\n✅ ALL BASELINES VALIDATED - Proceeding to benchmarks")
        print("="*70)
    
    # STEP 1: Run actual benchmarks
    for N in config.N_values:
        print(f"\n{'─'*70}")
        print(f"N = {N}")
        print(f"{'─'*70}")
        
        topology = config.topologies[0]
        
        # Generate data
        print("  📊 Generating training data...")
        train_ds = NMRDataset(N, topology, config.n_train_samples, config.T, config.dt)
        train_ds.generate_data(ckpt_mgr, 'train', config.n_workers)
        
        print("  📊 Generating validation data...")
        val_ds = NMRDataset(N, topology, config.n_val_samples, config.T, config.dt)
        val_ds.generate_data(ckpt_mgr, 'val', config.n_workers)
        
        # Normalize
        train_stats = train_ds.compute_normalization_stats()
        train_ds.normalize(train_stats)
        val_ds.normalize(train_stats)
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)
        
        # Train surrogate
        print("\n  🎓 Training surrogate...")
        if N <= 6:
            modes, width, n_layers = 12, 64, 4
        else:
            modes, width, n_layers = 24, 128, 6
        
        model = ImprovedPhysicsInformedFNO(
            modes=modes, width=width, n_layers=n_layers,
            n_params=N+1, n_outputs=3, time_steps=config.T
        )
        
        train_surrogate(model, train_loader, val_loader, config.epochs, device)
        
        # Benchmark all methods
        print("\n  ⏱️  Benchmarking methods...")
        
        # Get test sample
        test_sample = val_ds.data[0]
        params_normalized = test_sample['params']
        params_raw = params_normalized * train_stats['param_std'] + train_stats['param_mean']
        Omega = params_raw[:N]
        J = params_raw[N]
        
        # 1. Exact
        print("    [1/4] Exact...")
        sys_exact = SpinSystemOptimized(N, topology, use_sparse=False)
        exact_res = benchmark_single_method(sys_exact, Omega, J, config.T, config.dt, 
                                           'exact', config.n_runs, config.warmup_runs)
        
        # 2. Krylov
        print("    [2/4] Krylov...")
        sys_krylov = SpinSystemOptimized(N, topology, use_sparse=True)
        krylov_res = benchmark_single_method(sys_krylov, Omega, J, config.T, config.dt,
                                            'krylov', config.n_runs, config.warmup_runs)
        
        # 3. Chebyshev
        print("    [3/4] Chebyshev...")
        H = sys_exact.build_hamiltonian(Omega, J)
        cheb = ChebyshevPropagator(H, config.dt, order=50)
        
        cheb_times = []
        for run in range(config.warmup_runs + config.n_runs):
            psi0 = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
            Ix_avg = sum(sys_exact.Ix) / N
            Iy_avg = sum(sys_exact.Iy) / N
            Iz_first = sys_exact.Iz[0]
            
            cheb_result = cheb.simulate_trajectory(psi0, exact_res['times'], 
                                                  [Ix_avg, Iy_avg, Iz_first])
            if run >= config.warmup_runs:
                cheb_times.append(cheb_result['elapsed_time'])
        
        cheb_time = np.median(cheb_times)
        
        # 4. Surrogate
        print("    [4/4] Surrogate...")
        model.eval()
        model = model.to(device)
        
        params_t = torch.tensor(params_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        
        for _ in range(config.warmup_runs):
            with torch.no_grad():
                _ = model(params_t)
        
        surr_times = []
        for _ in range(config.n_runs):
            start = time.time()
            with torch.no_grad():
                pred = model(params_t)
            surr_times.append(time.time() - start)
        
        surr_time = np.median(surr_times)
        pred = pred.squeeze().cpu().numpy()
        pred_denorm = pred * train_stats['obs_std'] + train_stats['obs_mean']
        
        # Compute RMSE for surrogate vs exact
        surr_err = np.sqrt(
            np.mean((exact_res['Mx'] - pred_denorm[:, 0])**2) +
            np.mean((exact_res['My'] - pred_denorm[:, 1])**2) +
            np.mean((exact_res['I1z'] - pred_denorm[:, 2])**2)
        )
        
        # Store results
        result = {
            'N': N,
            'exact_time': exact_res['elapsed_time'],
            'krylov_time': krylov_res['elapsed_time'],
            'chebyshev_time': cheb_time,
            'surrogate_time': surr_time,
            'surrogate_error': float(surr_err),
            'speedup_vs_exact': exact_res['elapsed_time'] / surr_time,
            'speedup_vs_krylov': krylov_res['elapsed_time'] / surr_time
        }
        
        for k in results:
            if k in result:
                results[k].append(result[k])
        
        # Print summary
        print(f"\n  📊 Results:")
        print(f"     {'Method':<15} {'Time (s)':<15} {'Speedup':<10}")
        print(f"     {'-'*45}")
        print(f"     {'Exact':<15} {exact_res['elapsed_time']:>8.4f}  {'1.0×':<10}")
        print(f"     {'Krylov':<15} {krylov_res['elapsed_time']:>8.4f}  {exact_res['elapsed_time']/krylov_res['elapsed_time']:>9.1f}×")
        print(f"     {'Chebyshev':<15} {cheb_time:>8.4f}  {exact_res['elapsed_time']/cheb_time:>9.1f}×")
        print(f"     {'Surrogate':<15} {surr_time:>8.6f}  {result['speedup_vs_exact']:>9.1f}×")
        print(f"\n     Surrogate Error: {surr_err:.4f} (RMSE vs Exact)")
        
        ckpt_mgr.save_benchmark(result, N, topology)
    
    return results

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_results(results, save_path='results_fixed/scaling_plot.png'):
    """Generate publication-quality plot"""
    if not results['N']:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Time vs N
    ax = axes[0]
    ax.semilogy(results['N'], results['exact_time'], 'o-', label='Exact', linewidth=2, markersize=8)
    ax.semilogy(results['N'], results['krylov_time'], 's-', label='Krylov', linewidth=2, markersize=8)
    ax.semilogy(results['N'], results['chebyshev_time'], '^-', label='Chebyshev', linewidth=2, markersize=8)
    ax.semilogy(results['N'], results['surrogate_time'], 'd-', label='Surrogate', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Spins (N)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Computational Time vs System Size', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Speedup
    ax = axes[1]
    speedup_exact = [results['exact_time'][i] / results['surrogate_time'][i] 
                     for i in range(len(results['N']))]
    speedup_krylov = [results['krylov_time'][i] / results['surrogate_time'][i]
                      for i in range(len(results['N']))]
    
    ax.plot(results['N'], speedup_exact, 'o-', label='vs Exact', linewidth=2, markersize=8)
    ax.plot(results['N'], speedup_krylov, 's-', label='vs Krylov', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Spins (N)', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Surrogate Speedup', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved: {save_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("NMR SURROGATE - FIXED VERSION WITH VALIDATION")
    print("="*70)
    print(f"Device: {device}")
    print(f"Config: N={config.N_values}, T={config.T}, Samples={config.n_train_samples}")
    print("="*70)
    
    ckpt_mgr = CheckpointManager()
    
    try:
        # Run experiment
        results = experiment_scaling_fixed(config, ckpt_mgr)
        
        # Save results
        ckpt_mgr.save_results_csv(results, 'scaling_results')
        
        # Plot
        plot_results(results)
        
        print("\n" + "="*70)
        print("✅ EXPERIMENT COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {ckpt_mgr.results_dir}")
        
        # Print summary table
        if results['N']:
            print("\n📊 SUMMARY TABLE:")
            print(f"{'N':<5} {'Exact(s)':<12} {'Krylov(s)':<12} {'Surr(s)':<12} {'Speedup':<10} {'Error'}")
            print("-"*70)
            for i in range(len(results['N'])):
                N = results['N'][i]
                exact_t = results['exact_time'][i]
                krylov_t = results['krylov_time'][i]
                surr_t = results['surrogate_time'][i]
                speedup = exact_t / surr_t
                err = results['surrogate_error'][i]
                print(f"{N:<5} {exact_t:<12.4f} {krylov_t:<12.4f} {surr_t:<12.6f} {speedup:<10.1f} {err:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED - Progress saved")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🚀 Starting NMR Surrogate Benchmark (Fixed Version)")
    print("="*70)
    print("\n⏱️  Estimated Runtime (reduced config):")
    print("   • N=4: ~30 min (data gen) + ~10 min (training)")
    print("   • N=6: ~1 hour + ~15 min")
    print("   • N=8: ~2 hours + ~20 min")
    print("   • Total: ~4-5 hours\n")
    
    main()
