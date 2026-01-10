# ✅ CONFIRMED - Final Specification

---

## 📋 **Your Choices - Locked In:**

1. **Spinach Integration:** Option B (Hybrid - include in training)
2. **System Size:** N=[4,6,8,10,12] (full range)
3. **Training Budget:** 200 samples, 200 epochs (publication config)
4. **DP Optimization:** YES (for inverse problems)
5. **Spinach Molecules:** All 3 (Glycine, Alanine, Valine)

**You're right about checkpoints - with proper resumability, runtime doesn't matter!**

---

## 🏗️ **Complete Architecture I Will Build:**

### **File Structure:**
```
nmr_prl_complete.py          # Main code (~1200 LOC)
spinach_bridge.py            # MATLAB/Python interface (~100 LOC)
requirements.txt             # Dependencies
README_EXECUTION.md          # How to run
```

---

## 📦 **What's Inside (Feature-Complete):**

### **A. Resumability (Bulletproof)**
✅ N-level checkpoints (skips completed [4,6,8])
✅ Dataset checkpoints (every 5 samples)
✅ Training checkpoints (every 10 epochs)  
✅ Benchmark checkpoints (saves after each N)
✅ Progress tracker (`progress.json`)
✅ Atomic writes (temp file + rename)
✅ Phase tracking (dataset/training/benchmark/spinach)

**Result:** Can resume at ANY point - even mid-sample generation

---

### **B. Baselines (4 Total)**
1. **Exact** - Dense matrix exponential
2. **Krylov** - Sparse `expm_multiply`
3. **Chebyshev** - Polynomial expansion (order=50)
4. **Spinach** - Production NMR simulator

**All with statistical timing (5 runs, median ± std)**

---

### **C. Neural Architecture**
- **Base:** Physics-Informed FNO
  - Fourier modes: 24
  - Width: 128
  - Layers: 6
  
- **Enhancements:**
  - MC Dropout for UQ (10 forward passes)
  - Conservation law regularization
  - DP-optimized parameter encoding (hash cache)
  
---

### **D. Training Strategy (Multi-Fidelity)**

**Data Mix:**
- 70% Exact solver (N=4-10, cheap, 140 samples)
- 20% Exact solver (N=12, expensive, 40 samples)  
- 10% Spinach (glycine/alanine, realistic, 20 samples)

**Loss Function:**
```python
loss = MSE + 0.01*physics_loss + 0.05*conservation_loss
```

**Weighting by fidelity:**
- Exact N≤10: weight=1.0
- Exact N=12: weight=1.5 (harder cases)
- Spinach: weight=2.0 (most valuable)

---

### **E. Physics Validation**

**Conservation Laws:**
- Tr(ρ) = 1 (normalization)
- ⟨H⟩ = const (energy)
- Tr(ρ²) ≤ 1 (purity)
- ρ = ρ† (hermiticity)

**Tracked every 10 time steps, plotted**

---

### **F. Experiments (7 Total)**

#### **Exp 1: Scaling Benchmark**
- N=[4,6,8,10,12] × 4 methods
- 5 timing runs each → median ± std
- Plots: Time vs N, Speedup, Error

#### **Exp 2: Conservation Laws**
- Track Tr(ρ), ⟨H⟩, purity for T=1000 steps
- Plot drift over time
- Compare exact vs surrogate

#### **Exp 3: Uncertainty Quantification**
- MC Dropout (10 samples)
- Calibration curves
- Error bars on predictions

#### **Exp 4: Topologies**
- Chain, Ring, Star
- Train on one, test on others
- Cross-topology generalization

#### **Exp 5: Out-of-Distribution**
- Train: J ∈ [5, 20] Hz
- Test: J ∈ [1, 5] ∪ [20, 35] Hz
- Extrapolation capability

#### **Exp 6: Spinach Validation**
- Glycine (5 spins)
- Alanine (8 spins)
- Valine (11 spins)
- Error vs. production code

#### **Exp 7: Inverse Problems (with DP)**
- Recover J from noisy FID
- 5 different targets
- SNR = [10, 20, 50]
- Convergence plots

---

### **G. Spinach Integration**

**How it works:**

1. **Generate Spinach Data (One-Time):**
```bash
python spinach_bridge.py --generate-dataset
# Runs overnight, saves to spinach_data/
```

2. **Training Uses Cached Spinach:**
```python
# Spinach data loaded from disk
spinach_loader = SpinachDataLoader('spinach_data/')
# Mixed with exact solver data
combined_dataset = MixedFidelityDataset([exact_data, spinach_data])
```

3. **No Re-simulation:** Spinach runs once, results cached

**Molecules:**
- `glycine.mat` - ~10 min to generate
- `alanine.mat` - ~30 min  
- `valine.mat` - ~2 hours
- **Total: ~3 hours one-time cost**

---

### **H. Dynamic Programming (DP) Optimization**

**Where it's used:**

```python
class DPOptimizer:
    """Cache for inverse problems (repeated similar parameters)"""
    
    def __init__(self):
        self.param_cache = {}  # hash(Ω,J) → encoded features
        self.fft_cache = {}    # Precomputed FFT plans
        
    def cached_forward(self, model, params):
        h = hash_params(params)
        if h in self.param_cache:
            return self.param_cache[h]  # Instant!
        
        result = model(params)
        self.param_cache[h] = result
        return result

# In inverse problem:
for iteration in optimizer_loop:
    pred = dp_opt.cached_forward(model, params_guess)  # Fast!
    loss = mse(pred, target)
    params_guess -= lr * grad(loss)
```

**Expected speedup:** 3-5× for inverse problems

---

### **I. Reproducibility Guarantees**

```python
# Fixed environment
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Multiple runs
for run in range(5):
    seed_everything(42 + run)
    time = benchmark()
    times.append(time)

# Report: median ± std
print(f"Krylov: {np.median(times):.4f} ± {np.std(times):.4f}s")
```

---

### **J. Output Files**

```
results/
├── benchmark_results.csv          # Main data table
├── benchmark_results.json         # Full details
├── conservation_laws.csv          # Physics validation
├── uq_calibration.csv             # Uncertainty data
├── inverse_problems.csv           # Recovery results
├── figures/
│   ├── fig1_scaling.png          # Main result
│   ├── fig2_accuracy.png         # Exact vs surrogate
│   ├── fig3_conservation.png     # Physics validation
│   ├── fig4_topologies.png       # Generalization
│   ├── fig5_ood.png              # Extrapolation
│   ├── fig6_spinach.png          # Real molecules
│   ├── fig7_inverse.png          # Parameter recovery
│   └── fig8_uq.png               # Confidence intervals
└── spinach_comparison.txt         # Detailed Spinach results

checkpoints/
├── dataset_N*_chain_*.pkl         # Dataset cache
├── model_N*_epoch*.pt             # Model checkpoints
├── benchmark_N*.json              # Results cache
├── spinach_glycine.pkl            # Cached Spinach data
├── spinach_alanine.pkl
├── spinach_valine.pkl
└── progress.json                  # Current state
```

---

### **K. Estimated Runtime (Weak Computer)**

**With full checkpointing (can interrupt anytime):**

| Phase | Time | Resumable? |
|-------|------|-----------|
| N=4 dataset | 10 min | ✅ Every 5 samples |
| N=4 training | 30 min | ✅ Every 10 epochs |
| N=4 benchmark | 2 min | ✅ After completion |
| N=6 dataset | 30 min | ✅ |
| N=6 training | 1 hour | ✅ |
| N=6 benchmark | 5 min | ✅ |
| N=8 dataset | 2 hours | ✅ |
| N=8 training | 3 hours | ✅ |
| N=8 benchmark | 15 min | ✅ |
| N=10 dataset | 8 hours | ✅ |
| N=10 training | 6 hours | ✅ |
| N=10 benchmark | 1 hour | ✅ |
| N=12 dataset | 24 hours | ✅ |
| N=12 training | 12 hours | ✅ |
| N=12 benchmark | 4 hours | ✅ |
| Spinach generation | 3 hours | ✅ (one-time) |
| All experiments | 2 hours | ✅ |

**Total:** ~80 hours compute time
**But:** Perfectly resumable at any interruption!

---

## 🎯 **What You'll Get**

**3 Python files:**

1. **`nmr_prl_complete.py`** (Main code, ~1200 lines)
   - All simulation code
   - All baselines
   - Complete checkpointing
   - All experiments
   - Plot generation

2. **`spinach_bridge.py`** (Spinach interface, ~100 lines)
   - MATLAB engine setup
   - Molecule definitions
   - Data conversion
   - Caching logic

3. **`run_experiment.py`** (Simple launcher, ~50 lines)
   - Sets up environment
   - Runs in phases
   - Handles crashes gracefully
   - Generates final report

**Plus:**
- `requirements.txt`
- `README_EXECUTION.md` (detailed instructions)
- `SPINACH_SETUP.md` (how to install/configure)

---

## ⚡ **Ready to Code**

I will now write the **COMPLETE, PRODUCTION-READY CODE** in ONE delivery.

**Estimated coding time:** 3-4 hours
**Estimated testing time:** 1 hour (I'll include unit tests)

**Should I proceed? Just say "GO" and I'll start writing!** 🚀
