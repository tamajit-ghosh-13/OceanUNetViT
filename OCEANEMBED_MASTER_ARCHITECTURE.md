# 🌊 OceanEmbed: Technical Master Architecture & Physical Formulations
**3D Ocean Subsurface Thermal Inversion Engine for the North Indian Ocean**  
*Compiled: August 2026*

---

## 1. Executive Summary & Problem Formulation

The objective of **OceanEmbed** is to invert 2D satellite surface observations ($SST, SSS, SSH, \vec{u}, \vec{v}, \vec{w}$) into high-resolution, physically consistent 3D subsurface temperature volumes $T(x, y, z, t)$ across the North Indian Ocean from the sea surface down to $1000\,\text{m}$.

### Core Challenges Solved:
1. **The Thermocline Inversion Barrier ($50\,\text{m} - 150\,\text{m}$):** Intense temperature gradients ($\partial T / \partial z$) with sharp vertical curvature ($d^2T/dz^2$) that naive neural networks blur out.
2. **Extreme Non-Linear Climatic Dynamics:** Category 5 tropical cyclones (e.g. *Super Cyclone Amphan*), violent monsoon churn, and extreme Indian Ocean Dipole (IOD) upwelling/downwelling fronts.
3. **Hydrostatic Stability Preservation:** Enforcing physical stratification monotonicity ($T_{z+1} \le T_z$) so colder, denser water never floats artificially above warm mixed-layer water.

---

## 2. Geographic Domain & Coordinate Discretization

* **Geographic Bounding Box (North Indian Ocean):**
  $$\text{Latitude } \phi \in [5.0^\circ\text{N}, 30.0^\circ\text{N}], \quad \text{Longitude } \lambda \in [45.0^\circ\text{E}, 105.0^\circ\text{E}]$$
* **Horizontal Grid Resolution:** $\Delta \phi = 0.25^\circ, \Delta \lambda = 0.25^\circ \implies 101 \times 241\text{ spatial grid points}$.
* **Standard 15 Output Depth Levels ($z$ in meters):**
  $$z \in [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\,\text{m}$$

---

## 3. The 12 Input Surface Feature Channels & Exact Physics Formulations

The input tensor shape to the model is $(B, 12, 101, 241)$.

| Channel # | Name | Symbol | Units | Mathematical Formulation / Physical Meaning |
| :---: | :--- | :---: | :---: | :--- |
| **Ch 0** | Sea Surface Temperature | $\text{SST}$ | $^\circ\text{C}$ | Top thermal Dirichlet boundary condition at $z = 0\,\text{m}$. |
| **Ch 1** | Sea Surface Salinity | $\text{SSS}$ | $\text{PSU}$ | Surface haline boundary condition affecting upper water mass buoyancy. |
| **Ch 2** | Sea Surface Height | $\text{SSH}$ | $\text{m}$ | Altimetric sea level anomaly; drives baroclinic thermocline displacement. |
| **Ch 3** | Surface Zonal Current | $u_{\text{cur}}$ | $\text{m/s}$ | East-West ocean surface advection velocity ($u$). |
| **Ch 4** | Surface Meridional Current | $v_{\text{cur}}$ | $\text{m/s}$ | North-South ocean surface advection velocity ($v$). |
| **Ch 5** | Zonal Wind Stress | $u_{\text{wind}}$ | $\text{m/s}$ | Geostrophically balanced atmospheric wind component: $$u_{\text{wind}} = -\frac{g}{f} \frac{\partial \eta}{\partial y}$$ |
| **Ch 6** | Meridional Wind Stress | $v_{\text{wind}}$ | $\text{m/s}$ | Geostrophically balanced atmospheric wind component: $$v_{\text{wind}} = \frac{g}{f} \frac{\partial \eta}{\partial x}$$ |
| **Ch 7** | Mechanical Wind Magnitude | $|\vec{w}|$ | $\text{m/s}$ | Total surface wind shear energy driving mixed layer deepening: $$|\vec{w}| = \sqrt{u_{\text{wind}}^2 + v_{\text{wind}}^2}$$ |
| **Ch 8** | Seasonal Harmonic Sin | $\sin(\theta_{\text{DOY}})$ | $[-1, 1]$ | Annual solar radiation cycle: $$\sin\left(\frac{2\pi \cdot \text{DOY}}{365}\right)$$ |
| **Ch 9** | Seasonal Harmonic Cos | $\cos(\theta_{\text{DOY}})$ | $[-1, 1]$ | Annual monsoon phase cycle: $$\cos\left(\frac{2\pi \cdot \text{DOY}}{365}\right)$$ |
| **Ch 10** | Climatological SST Anomaly | $\text{SST}_{\text{anom}}$ | $^\circ\text{C}$ | Deviation from temporal climatology: $$\text{SST}_{\text{anom}}(x,y,t) = \text{SST}(x,y,t) - \overline{\text{SST}}(x,y)$$ |
| **Ch 11** | Potential Density Anomaly | $\sigma_0$ | $\text{kg/m}^3$ | TEOS-10 Equation of Seawater: $$\sigma_0 = \rho(S_A, \Theta, 0) - 1000\,\text{kg/m}^3$$ |

> **Coriolis Parameter Stabilization ($f$):**  
> $$f = 2\Omega \sin\phi, \quad \Omega = 7.2921 \times 10^{-5}\,\text{rad/s}$$  
> Near the equator ($|\phi| < 5^\circ$), $f$ is stabilized with a physical threshold $|f| \ge 10^{-5}\,\text{s}^{-1}$ to eliminate mathematical singularities in wind curl computations.

---

## 4. Deep Neural Architecture: `OceanUNetViT`

A hybrid **Convolutional Vision Transformer (U-Net ViT)** combining multi-scale spatial CNN convolutions with global attention teleconnections:

```
[Input: (B, 12, 101, 241)]
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4-LEVEL CNN ENCODER                                         │
│   Level 1: (B,  64, 101, 241) ── Skip Connection 1 ───────┐ │
│   Level 2: (B, 128,  50, 120) ── Skip Connection 2 ─────┐ │ │
│   Level 3: (B, 256,  25,  60) ── Skip Connection 3 ───┐ │ │ │
│   Level 4: (B, 512,  12,  30) ── Skip Connection 4 ─┐ │ │ │ │
└──────────────────────────────┬──────────────────────│─│─│─│─┘
                               │                      │ │ │ │
                               ▼                      │ │ │ │
┌───────────────────────────────────────────────────┐ │ │ │ │
│ 8-HEAD VISION TRANSFORMER BOTTLENECK              │ │ │ │ │
│   Tokens: 360 (12x30), d_embed = 256              │ │ │ │ │
│   Basin-scale teleconnections (Arabian Sea <-> BoB)│ │ │ │ │
└──────────────────────────────┬────────────────────┘ │ │ │ │
                               │                      │ │ │ │
                               ▼                      │ │ │ │
┌─────────────────────────────────────────────────────│─│─│─│─┐
│ 4-LEVEL CNN DECODER                                 │ │ │ │ │
│   Level 4: (B, 256,  25,  60) ◀── Concat Skip 4 ────┘ │ │ │ │
│   Level 3: (B, 128,  50, 120) ◀── Concat Skip 3 ──────┘ │ │ │
│   Level 2: (B,  64, 101, 241) ◀── Concat Skip 2 ────────┘ │ │
│   Level 1: (B,  64, 101, 241) ◀── Concat Skip 1 ──────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1x1 CONVOLUTION + CLIMATOLOGICAL DEPTH BIAS VECTOR          │
│   b_z = [28.5, 28.3, 28.1, 27.5, 26.5, 24.0, ... 5.0]°C     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
[Output 3D Volume: (B, 15, 101, 241)]
```

* **DoubleConv Unit:** $\text{Conv2D}(3\times 3, p=1) \to \text{BatchNorm2D} \to \text{GELU} \to \text{Conv2D}(3\times 3, p=1) \to \text{BatchNorm2D} \to \text{GELU}$.
* **Depth Bias $\vec{b} \in \mathbb{R}^{15}$:** Allows the network to learn **residual thermal perturbations** $\Delta T(x,y,z,t)$ rather than predicting absolute physical degrees from scratch.

---

## 5. Physics-Informed Loss Formulation

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{recon}} + \lambda_{\text{grad}} \cdot \mathcal{L}_{\text{grad}} + \lambda_{\text{curv}} \cdot \mathcal{L}_{\text{curv}} + \lambda_{\text{mono}} \cdot \mathcal{L}_{\text{mono}}$$

1. **Layer-Weighted Thermocline Reconstruction Loss ($\mathcal{L}_{\text{recon}}$):**
   $$\mathcal{L}_{\text{recon}} = \frac{1}{BHW} \sum_{b,h,w} \sum_{z=1}^{15} w(z) \cdot \left( \hat{T}(b,z,h,w) - T(b,z,h,w) \right)^2$$
   $$w(z) = [1.0, 1.0, 1.0, 1.2, 1.5, \mathbf{2.5, 3.0, 3.5, 3.0, 2.5}, 1.8, 1.2, 1.0, 1.0, 1.0]$$
2. **Vertical Temperature Gradient Loss ($\mathcal{L}_{\text{grad}}$):**
   $$\mathcal{L}_{\text{grad}} = \frac{1}{14} \sum_{z=1}^{14} \left( \frac{\hat{T}_{z+1} - \hat{T}_z}{\Delta z} - \frac{T_{z+1} - T_z}{\Delta z} \right)^2$$
3. **Vertical Curvature Loss ($\mathcal{L}_{\text{curv}}$):**
   $$\mathcal{L}_{\text{curv}} = \frac{1}{13} \sum_{z=1}^{13} \left( \nabla_z^2 \hat{T}_z - \nabla_z^2 T_z \right)^2$$
4. **Stratification Monotonicity Constraint ($\mathcal{L}_{\text{mono}}$):**
   $$\mathcal{L}_{\text{mono}} = \frac{1}{14} \sum_{z=1}^{14} \text{ReLU}\left( \hat{T}_{z+1} - \hat{T}_z \right)$$

---

## 6. Model Evolutionary Spectrum & Datasets

### 1. Baseline Model (7 Channels)
* **Dataset:** 2025–2026 satellite split (273 Days).
* **Loss:** Unweighted MSE.
* **Limitation:** Severely blurred the thermocline; average RMSE was **$1.0452^\circ\text{C}$**.

### 2. OceanUNetViT v3 Unbiased (12 Channels)
* **Dataset:** 2023–2024 Multi-Season Catalog (457 Days).
* **Innovations:** Added 5 thermodynamic channels (Wind magnitude, Seasonal harmonics, SST anomaly, TEOS-10 $\sigma_0$); depth-wise standardized target space $(T_z - \mu_z)/\sigma_z$.
* **RMSE:** **$0.9584^\circ\text{C}$**.

### 3. OceanUNetViT v4 (Physics-Guided Stratification)
* **Dataset:** Multi-Season GLORYS12.
* **Innovations:** Introduced the 4-part physics loss with layer-weighting and monotonicity penalties.
* **RMSE:** **$0.9149^\circ\text{C}$**.

### 4. OceanUNetViT v4_extended (12-Month Extreme Anomaly Training) 🚀
* **Dataset:** 12 distinct multi-decade extreme climatic anomaly months:
  1. *Nov 2016:* Record Super Negative IOD (-IOD)
  2. *Jun 2020:* Super Cyclone Amphan category 5 mixing
  3. *Oct 2019:* Record Super Positive IOD (+IOD)
  4. *May 2021:* Cyclones Tauktae & Yaas pre-monsoon churn
  5. *Dec 2015:* Super El Niño Basin-Wide Heat Anomaly
  6. *May 2010:* Pre-Monsoon Warm Pool Thermal Peak
  7. *Aug 2018:* Kerala Super Monsoon & Somali Jet Upwelling
  8. *May 2019:* Super Cyclone Fani
  9. *Nov 2017:* Cyclone Ockhi rapid intensification
  10. *Aug 2008:* Strong Positive IOD Summer Upwelling
  11. *Jul 2013:* Southwest Monsoon deep active phase
  12. *Jan 2012:* La Niña winter Northeast Monsoon convection
* **Standalone Benchmark:** **$0.7686^\circ\text{C}$ ($r=0.902$)** across 519,519 in-situ float measurements.

### 5. OceanUNetViT v5_finetuned (In-Situ Inversion Calibration)
* **Dataset:** Unified dataset of **792,481 in-situ physical ARGO float observations** + GLORYS reanalysis.
* **Specialty:** Dynamic Isotherm Pinning ($D_{20}$) and deep-sea hydrostatic anchoring ($200\,\text{m} - 1000\,\text{m}$).

---

## 7. The Breakthrough: Duo-Elite Ensemble Formulation

### Why Previous Ensembles (Tri-Breed / Quad-Breed) Underperformed:
The legacy Tri-Breed ensemble combined Baseline (7ch), v3 (12ch), and early v4. Because Baseline and v3 had errors exceeding $1.0^\circ\text{C}$, blending them **diluted and dragged down** the superior performance of `v4_extended`.

### The Solution: Sequential Least Squares Programming (SLSQP) Simplex Optimization
We solved depth-by-depth optimal convex weights pairing **`v4_extended`** and **`v5_finetuned`**:

$$\min_{\vec{w}_{v4}, \vec{w}_{v5}} \sum_{i=1}^{N_{\text{obs}}} \left( w_{v4}(z) \cdot \hat{T}_{v4\_ext}^i(z) + w_{v5}(z) \cdot \hat{T}_{v5\_ft}^i(z) - T_{\text{Argo}}^i(z) \right)^2 \quad \text{s.t.} \quad w_{v4}(z) + w_{v5}(z) = 1.0, \quad w \ge 0$$

```python
DUO_ELITE_WEIGHTS = {
    0:    [0.5000, 0.5000],  # 0m:   50% v4_ext / 50% v5_ft  (Equal SST anchoring)
    5:    [0.5961, 0.4039],  # 5m:   60% v4_ext / 40% v5_ft  (Surface turbulence synergy)
    10:   [0.7996, 0.2004],  # 10m:  80% v4_ext / 20% v5_ft  (Upper mixed-layer shear)
    20:   [0.8224, 0.1776],  # 20m:  82% v4_ext / 18% v5_ft  (Mixed layer depth base)
    30:   [0.8563, 0.1437],  # 30m:  86% v4_ext / 14% v5_ft  (Thermocline transition)
    50:   [0.8605, 0.1395],  # 50m:  86% v4_ext / 14% v5_ft  (Core thermocline top)
    75:   [0.8607, 0.1393],  # 75m:  86% v4_ext / 14% v5_ft  (Upwelling front tracking)
    100:  [0.6812, 0.3188],  # 100m: 68% v4_ext / 32% v5_ft  (Isotherm transition zone)
    125:  [0.9199, 0.0801],  # 125m: 92% v4_ext /  8% v5_ft  (Deep thermocline boundary)
    150:  [0.9070, 0.0930],  # 150m: 91% v4_ext /  9% v5_ft  (Sub-thermocline continuity)
    200:  [0.4962, 0.5038],  # 200m: 50% v4_ext / 50% v5_ft  (Intermediate water crossover)
    300:  [0.0441, 0.9559],  # 300m:  4% v4_ext / 96% v5_ft  (v5 Hydrostatic density pinning)
    500:  [0.2275, 0.7725],  # 500m: 23% v4_ext / 77% v5_ft  (Deep intermediate stability)
    700:  [0.3026, 0.6974],  # 700m: 30% v4_ext / 70% v5_ft  (Antarctic Intermediate Water)
    1000: [0.2169, 0.7831],  # 1000m:22% v4_ext / 78% v5_ft  (Abyssal thermal boundary)
}
```

---

## 8. Master In-Situ Ground Truth Validation Benchmark (668,045 Float Observations)

Every prediction was evaluated against physical **Ifremer GDAC ARGO CTD Float Sensors** using continuous sub-grid **2D bilinear + vertical PCHIP splining**:

| Historical Climatic Test Era | Physical Float Obs | Baseline (7-ch) | Standalone `v4_extended` | **Duo-Elite Ensemble 🏆** | Error Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **November 2016 (Historic -IOD)** | 152,981 | $1.0583^\circ\text{C}$ | $0.7115^\circ\text{C}$ | **$0.6765^\circ\text{C}$ ($r=0.920$)** | **$-36.1\%$** |
| **June 2020 (Super Cyclone Amphan)** | 104,617 | $1.0983^\circ\text{C}$ | $0.7052^\circ\text{C}$ | **$0.6844^\circ\text{C}$ ($r=0.885$)** | **$-37.7\%$** |
| **October 2019 (Historic +IOD)** | 121,145 | $1.0641^\circ\text{C}$ | $0.8687^\circ\text{C}$ | **$0.7906^\circ\text{C}$ ($r=0.921$)** | **$-25.7\%$** |
| **May 2021 (Cyclones Tauktae/Yaas)**| 140,776 | $0.9841^\circ\text{C}$ | $0.8092^\circ\text{C}$ | **$0.6764^\circ\text{C}$ ($r=0.900$)** | **$-31.3\%$** |
| **April 2018 (Warm Pool Peak)** | 148,526 | $\sim 1.08^\circ\text{C}$ | $0.8786^\circ\text{C}$ | **$0.8036^\circ\text{C}$ ($r=0.886$)** | **$-25.6\%$** |
| **GRAND TOTAL** | **668,045** | **$1.0520^\circ\text{C}$** | **$0.7930^\circ\text{C}$** | **$0.7285^\circ\text{C}$ ($r=0.903$) 🏆** | **$\mathbf{30.8\%}$ Global Drop** |

---

## 9. Key Directory Layout

* `config.py` — Central single source of truth for grid parameters, 12 channels, and 15 depths.
* `model.py` — Hybrid `OceanUNetViT` (DoubleConv + 8-Head ViT + DepthBias).
* `scripts/train_v4_extended.py` — 12-Month extreme anomaly training pipeline.
* `scripts/train_v5.py` — Physics-preserving thermocline loss training.
* `scripts/solve_duo_elite_ensemble.py` — SLSQP optimal simplex weight solver.
* `scripts/evaluate_duo_elite_ensemble.py` — Multi-era ARGO ground truth validation benchmark.
* `scripts/evaluate_fresh_argo_duo_elite.py` — On-demand fresh ERDDAP float extraction and validation.
* `checkpoints/best_ocean_model_v4_extended.pt` — SOTA anomaly-trained model weights.
* `checkpoints/best_ocean_model_v5_finetuned.pt` — SOTA deep-anchored in-situ model weights.