# 🌊 OceanEmbed: 3D Ocean Temperature Inversion

A PyTorch deep learning framework to reconstruct full **3D subsurface ocean temperature profiles** (0m to 1000m) from **2D surface satellite observations** (Sea Surface Temperature, Salinity, and Height), specifically optimized for **Apple Silicon (M4 / MPS)**.

---

## 🎯 What This Project Does

Satellites can only "see" the top millimeter of the ocean (the surface). However, deep ocean dynamics, marine heatwaves, and subsurface currents require knowing temperature at varying depths (0m to 1000m). 

**OceanEmbed** learns the non-linear physical relationship between 2D surface variables and 3D subsurface temperature fields using a **Hybrid U-Net + Vision Transformer (ViT)** neural network.

---

## 📁 Project Structure

```
OceanEmbed/
├── requirements.txt         # List of Python dependencies (PyTorch, Copernicus Marine, xarray)
├── data_loader.py           # Copernicus streaming & PyTorch Dataset with spatial subsetting
├── model.py                 # Hybrid U-Net + Vision Transformer neural network
├── train.py                 # Training script with Apple Silicon MPS acceleration & safety guards
└── README.md                # This step-by-step mentor guide
```

### File Explanations:
1. **`requirements.txt`**: Contains all library specifications including PyTorch, xarray, and the Copernicus Marine client.
2. **`data_loader.py`**: Handles lazy streaming and spatial bounding-box subsetting from Copernicus servers so you never need to download huge global datasets. Includes a mock generator for instant offline testing.
3. **`model.py`**: Defines the `OceanUNetViT` neural network:
   - **Encoder (CNN)**: Extracts local ocean eddies and coastal boundaries.
   - **Bottleneck (ViT Self-Attention)**: Learns large-scale basin teleconnections across the North Indian Ocean.
   - **Decoder (Transposed CNN)**: Upsamples features and projects them into 14 vertical depth levels.
4. **`train.py`**: Configures the training loop, sets up Apple Silicon MPS (`torch.device("mps")`), and includes safety pause guards.

---

## 🗺️ Target Region: North Indian Ocean

To keep memory and storage footprint tiny, the pipeline crops remote data server-side to the North Indian Ocean bounding box:
- **Latitude**: `5.0°N` to `30.0°N` (Arabian Sea & Bay of Bengal)
- **Longitude**: `45.0°E` to `105.0°E` (Horn of Africa to Malacca Strait)

---

## ⚡ Apple Silicon (M4 / MPS) Acceleration

PyTorch uses Apple's **Metal Performance Shaders (MPS)** to run tensor matrix multiplications directly on your Mac GPU:

```python
import torch

# Automatic device selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## 🚀 Step-by-Step Instructions

### Step 1: Install Dependencies
Open your terminal in this directory and run:

```bash
pip install -r requirements.txt
```

---

### Step 2: Test the Data Pipeline (Offline / Mock Mode)
Run a quick self-test of `data_loader.py`:

```bash
python3 data_loader.py
```

You will see:
```
Input Surface Tensor Shape:   torch.Size([3, 64, 128])
Target 3D Subsurface Shape:   torch.Size([14, 64, 128])
```

---

### Step 3: Test the Hybrid U-Net + ViT Architecture
Run a model forward-pass verification on your Mac GPU:

```bash
python3 model.py
```

You will see the model instantiated with **678,062 parameters** running on device `mps`.

---

### Step 4: Run the Training Pre-Flight Check
Run `train.py` to verify the pipeline:

```bash
python3 train.py
```

This performs a dry-run check and safely pauses before training.

---

### Step 5: (Optional) Connect Copernicus Marine Account
When you are ready to use live satellite data instead of mock data:
1. Create a free account at [marine.copernicus.eu](https://marine.copernicus.eu/).
2. In your terminal or Python code, authenticate:
   ```bash
   copernicusmarine login --username <YOUR_USERNAME> --password <YOUR_PASSWORD>
   ```

---

### Step 6: Launch Actual Training Loop
When you are ready to train the model over full epochs:

```bash
python3 -c "import train; train.run_training_experiment(epochs=20)"
```
Checkpoints will be saved automatically to `./checkpoints/best_ocean_model.pt`.
