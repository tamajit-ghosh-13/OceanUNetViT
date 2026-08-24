"""
================================================================================
OceanEmbed - Dual-Branch Sparse ARGO Graph Fusion Network (argo_gnn_fusion.py)
================================================================================
Implements Feature #6:
  Branch A: Dense Satellite 2D Convolutional Encoder (101x241 grid)
  Branch B: Sparse In-Situ ARGO Observation Graph Network (K-Nearest Float Nodes)
  Bottleneck: Cross-Attention Multi-Modal Fusion Layer

Where ARGO float in-situ truth is available, the network anchors and refines
surrounding satellite predictions.
================================================================================
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
)
from model import create_model, EncoderBlock, DoubleConv
from train import get_compute_device


class SparseArgoGraphLayer(nn.Module):
    """
    Graph Neural Layer operating on sparse in-situ float coordinates.
    Node features: [lat, lon, depth_profile_15] (17-dim)
    """
    def __init__(self, in_features: int = 17, hidden_dim: int = 128):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_feats: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_feats: (N_floats, in_features)
        adj_matrix: (N_floats, N_floats) - Gaussian distance kernel
        """
        H = self.node_mlp(node_feats) # (N, hidden_dim)
        # Message passing with adjacency weighting
        messages = torch.matmul(adj_matrix, H) # (N, hidden_dim)
        return H + messages


class DualBranchSatelliteArgoFusionNet(nn.Module):
    """
    Fuses dense satellite feature maps with sparse in-situ Argo float graph nodes
    via spatial Cross-Attention.
    """
    def __init__(self, in_channels: int = 12, out_depth_levels: int = 15):
        super().__init__()
        # Dense Satellite Encoder
        self.sat_enc1 = EncoderBlock(in_channels, 64)
        self.sat_enc2 = EncoderBlock(64, 128)
        self.sat_enc3 = EncoderBlock(128, 256)
        self.sat_enc4 = EncoderBlock(256, 512)

        # Sparse Argo Graph Branch
        self.gnn = SparseArgoGraphLayer(in_features=17, hidden_dim=512)

        # Cross-Attention Fusion (Dense Satellite Query <-> Sparse Argo Keys/Values)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        # Output Decoder
        self.dec_conv = nn.Sequential(
            DoubleConv(512, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(128, 64),
            nn.Upsample(size=(101, 241), mode="bilinear", align_corners=True),
            nn.Conv2d(64, out_depth_levels, kernel_size=1)
        )

    def forward(
        self,
        sat_inputs: torch.Tensor,
        argo_nodes: torch.Tensor = None,
        argo_adj: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. Satellite Dense Encoding
        s1, p1 = self.sat_enc1(sat_inputs)
        s2, p2 = self.sat_enc2(p1)
        s3, p3 = self.sat_enc3(p2)
        s4, p4 = self.sat_enc4(p3) # (B, 512, 12, 30)

        B, C, H, W = p4.shape
        sat_tokens = p4.flatten(2).transpose(1, 2) # (B, 360, 512)

        if argo_nodes is not None and argo_adj is not None:
            # 2. Graph Message Passing
            float_feats = self.gnn(argo_nodes, argo_adj).unsqueeze(0).expand(B, -1, -1) # (B, N_floats, 512)
            # 3. Cross-Attention: Satellite tokens query Sparse Argo float nodes
            fused_tokens, _ = self.cross_attn(query=sat_tokens, key=float_feats, value=float_feats)
            sat_tokens = sat_tokens + fused_tokens

        fused_spatial = sat_tokens.transpose(1, 2).reshape(B, C, H, W)
        predictions = self.dec_conv(fused_spatial) # (B, 15, 101, 241)
        return predictions


def test_argo_gnn_fusion():
    print("\n" + "=" * 105)
    print("🛸 TESTING DUAL-BRANCH SATELLITE + SPARSE ARGO GRAPH NEURAL FUSION NETWORK")
    print("=" * 105)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DualBranchSatelliteArgoFusionNet(in_channels=12, out_depth_levels=15).to(device)

    # 1. Dummy Satellite Tensor (Batch=1, 12 channels, 101, 241)
    dummy_sat = torch.randn(1, 12, 101, 241, device=device)

    # 2. Dummy Sparse Argo Observations (N=24 active float buoys across the basin)
    dummy_argo_nodes = torch.randn(24, 17, device=device) # lat, lon + 15 temps
    # Distance-based adjacency matrix
    dummy_adj = torch.softmax(torch.randn(24, 24, device=device), dim=-1)

    with torch.no_grad():
        out_fused = model(dummy_sat, dummy_argo_nodes, dummy_adj)

    print(f"   Satellite Dense Grid: {tuple(dummy_sat.shape)}")
    print(f"   Sparse Float Graph:   {tuple(dummy_argo_nodes.shape)} nodes ({dummy_argo_nodes.shape[0]} buoys)")
    print(f"   Fused Output Volume:  {tuple(out_fused.shape)} (15 Depth Levels)")
    print(f"   Total GNN-Fusion Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Feature #6 (Dual Encoder Satellite + ARGO Graph Fusion) Successfully Architected!")


if __name__ == "__main__":
    test_argo_gnn_fusion()
