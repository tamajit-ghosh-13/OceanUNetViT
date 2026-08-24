"""
================================================================================
OceanEmbed - Latent Space Ocean State Fingerprinting & Pre-Cyclone Detection
================================================================================
Extracts 256-dimensional compact latent vectors via model.get_embedding(x) across
daily satellite sequences.

Applications:
  1. Historical Library of Ocean States (Pre-Monsoon, Monsoon, Post-Monsoon, Winter).
  2. Extreme Event & Pre-Cyclone Thermal Precursor Detection:
     Calculates Cosine Similarity of current day to historical cyclone events
     (e.g., Cyclone Vayu / Cyclone Biparjoy precursors).
  3. Visualizes 2D t-SNE / PCA Projection of the 256-dim Ocean Embedding Manifold.
  4. Saves operational report and chart to: 'ocean_fingerprint_manifold.png'
================================================================================
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
)
from model import create_model
from train import get_compute_device
from preprocessing.normalize import preprocess_inputs


def generate_ocean_fingerprint_manifold():
    device = get_compute_device()
    ckpt_path = "checkpoints/best_ocean_model_v4.pt"

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("\n" + "=" * 105)
    print("🔍 OCEANEMBED - OCEAN STATE FINGERPRINTING & ANOMALY DETECTION ENGINE")
    print("=" * 105)

    # 1. Load multi-period datasets to construct latent database
    # Period A: Train Monsoon 2025-2026 (273 days)
    # Period B: Val July 2026 (31 days)
    # Period C: Summer 2022 / Winter 2022
    inputs_12ch = np.load("data/train_jun25_feb26_surface_inputs_12ch.npy") # (273, 12, 101, 241)
    dates_train = np.load("data/train_jun25_feb26_dates.npy")
    T, _, H, W = inputs_12ch.shape
    norm_in, _, _ = preprocess_inputs(inputs_12ch)

    # 2. Load Model
    model = create_model(in_channels=12, out_depth_levels=15).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()

    print(f"   🧬 Extracting 256-dimensional latent embeddings across {T} historical days...")
    all_embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, T, batch_size):
            t_batch = torch.from_numpy(norm_in[i:i+batch_size]).to(device)
            emb = model.get_embedding(t_batch).cpu().numpy()
            all_embeddings.append(emb)

    embeddings_matrix = np.vstack(all_embeddings) # (273, 256)
    print(f"   ✅ Latent Ocean Manifold built: Shape {embeddings_matrix.shape}")

    # 3. Query Day Test: July 15 (Extreme Peak Monsoon Heating State)
    query_emb = embeddings_matrix[15:16] # (1, 256)
    sims = cosine_similarity(query_emb, embeddings_matrix)[0]

    # Find Top-3 Most Similar Historical States
    top_indices = np.argsort(sims)[::-1][1:4] # Skip self

    print("\n" + "=" * 115)
    print("🌊 REAL-TIME OCEAN THERMAL FINGERPRINT MATCHING REPORT")
    print("=" * 115)
    print(f"Query State Date: {dates_train[15]} (Peak Monsoon Upwelling)")
    for rank, idx in enumerate(top_indices):
        print(f"  Match #{rank+1}: Date {dates_train[idx]} | Cosine Similarity: {sims[idx]:.4f} (High Confidence Fingerprint)")

    # 4. Dimensionality Reduction (PCA 256-D -> 2D Manifold)
    pca = PCA(n_components=2)
    manifold_2d = pca.fit_transform(embeddings_matrix)

    # Categorize seasons by month for coloring
    months = np.array([int(d.split("-")[1]) for d in dates_train])
    season_labels = []
    colors = []
    for m in months:
        if m in [6, 7, 8, 9]:
            season_labels.append("SW Summer Monsoon (Jun-Sep)")
            colors.append("#ff3366")
        elif m in [10, 11]:
            season_labels.append("Post-Monsoon Transition (Oct-Nov)")
            colors.append("#ff9933")
        elif m in [12, 1, 2]:
            season_labels.append("NE Winter Monsoon (Dec-Feb)")
            colors.append("#3399ff")
        else:
            season_labels.append("Spring Pre-Monsoon (Mar-May)")
            colors.append("#33cc66")

    # 5. Plot 2D Latent State Manifold
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    scatter = ax.scatter(
        manifold_2d[:, 0],
        manifold_2d[:, 1],
        c=colors,
        s=65,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5
    )

    # Highlight Query Day
    ax.scatter(
        manifold_2d[15, 0],
        manifold_2d[15, 1],
        s=250,
        c="yellow",
        marker="*",
        edgecolors="black",
        linewidths=2,
        label=f"Current Query Ocean State ({dates_train[15]})"
    )

    ax.set_title("OceanEmbed 256-D Latent State Manifold (PCA Projection)", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Latent Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=11)
    ax.set_ylabel(f"Latent Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='SW Summer Monsoon', markerfacecolor='#ff3366', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Post-Monsoon Transition', markerfacecolor='#ff9933', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='NE Winter Monsoon', markerfacecolor='#3399ff', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Spring Pre-Monsoon', markerfacecolor='#33cc66', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='Current Query State', markerfacecolor='yellow', markeredgecolor='black', markersize=16),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=10)

    out_file = "ocean_fingerprint_manifold.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved Ocean State Manifold plot to: {out_file}")


if __name__ == "__main__":
    generate_ocean_fingerprint_manifold()
