#!/usr/bin/env python3
"""
analyse_multiome.py - Analyze multiome data (RNA + ATAC) with joint integration
Usage: python analyse_multiome.py -i filtered.h5mu -o analyzed.h5mu
"""

import argparse
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze multiome data (RNA + ATAC)')
    parser.add_argument('-i', '--input', required=True, help='Input h5mu file (filtered)')
    parser.add_argument('-o', '--output', required=True, help='Output h5mu file')
    return parser.parse_args()

def tfidf_transform(adata, scale_factor=1e4):
    """Manual TF-IDF transformation for ATAC."""
    from scipy import sparse
    
    if sparse.issparse(adata.X):
        X = adata.X.tocsr()
    else:
        X = adata.X
    
    n_cells = X.shape[0]
    idf = np.log(1 + n_cells / (1 + (X > 0).sum(axis=0)))
    
    if sparse.issparse(X):
        idf_diag = sparse.diags(idf.A1 if hasattr(idf, 'A1') else idf.flatten())
        adata.X = X.dot(idf_diag) * scale_factor
    else:
        adata.X = X * idf * scale_factor
    
    return adata

def main():
    args = parse_args()
    
    base_prefix = os.path.splitext(os.path.basename(args.input))[0]
    
    print(f"Loading filtered data from {args.input}")
    mdata = mu.read_h5mu(args.input)
    
    # Check what modalities we have
    print(f"Available modalities: {list(mdata.mod.keys())}")
    
    # Process RNA data
    print("\n=== PROCESSING RNA DATA ===")
    rna = mdata.mod['rna']
    print(f"RNA shape: {rna.shape}")
    
    # RNA normalization
    print("\n1. Normalizing RNA data...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    
    # Add RNA QC metrics if not present
    if 'n_counts' not in rna.obs.columns:
        rna.obs['n_counts'] = rna.X.sum(axis=1).A1 if hasattr(rna.X, 'A1') else rna.X.sum(axis=1)
    if 'n_genes' not in rna.obs.columns:
        rna.obs['n_genes'] = (rna.X > 0).sum(axis=1).A1 if hasattr(rna.X, 'A1') else (rna.X > 0).sum(axis=1)
    
    print(f"RNA obs columns after QC: {list(rna.obs.columns)}")
    
    # RNA feature selection
    print("2. Selecting highly variable genes...")
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor='cell_ranger')
    
    # RNA PCA
    print("3. Running PCA on RNA...")
    rna_hv = rna[:, rna.var['highly_variable']].copy()
    sc.pp.scale(rna_hv, max_value=10)
    sc.tl.pca(rna_hv, n_comps=50, random_state=42)
    
    # Copy PCA to main RNA object
    rna.obsm['X_pca'] = rna_hv.obsm['X_pca']
    rna.uns['pca'] = rna_hv.uns['pca']
    
    # Process ATAC data
    print("\n=== PROCESSING ATAC DATA ===")
    atac = mdata.mod['atac']
    print(f"ATAC shape: {atac.shape}")
    
    # ATAC normalization (TF-IDF)
    print("\n1. Performing TF-IDF normalization...")
    sc.pp.normalize_total(atac, target_sum=1e4)
    atac = tfidf_transform(atac, scale_factor=1e4)
    
    # ATAC feature selection
    print("2. Selecting highly variable peaks...")
    # Simple variance-based feature selection for ATAC
    from scipy import sparse
    if sparse.issparse(atac.X):
        X = atac.X.tocsr()
    else:
        X = atac.X
    
    # Calculate variance
    mean = np.array(X.mean(axis=0)).flatten()
    var = np.array(X.power(2).mean(axis=0)).flatten() - mean ** 2
    
    # Select top 25k features by variance
    n_features = min(25000, atac.n_vars)
    top_idx = np.argsort(var)[-n_features:]
    
    high_var_mask = np.zeros(atac.n_vars, dtype=bool)
    high_var_mask[top_idx] = True
    atac.var['highly_variable'] = high_var_mask
    
    # ATAC PCA (LSI)
    print("3. Running PCA on ATAC (LSI)...")
    atac_hv = atac[:, atac.var['highly_variable']].copy()
    sc.pp.scale(atac_hv, max_value=10)
    sc.tl.pca(atac_hv, n_comps=50, random_state=42)
    
    # Copy PCA to main ATAC object
    atac.obsm['X_lsi'] = atac_hv.obsm['X_pca']
    atac.uns['lsi'] = atac_hv.uns['pca']
    
    # Update mdata with processed modalities
    mdata.mod['rna'] = rna
    mdata.mod['atac'] = atac
    
    # === INTEGRATION ===
    print("\n=== INTEGRATING RNA AND ATAC ===")
    
    # Make sure we're using the same cells in both modalities
    print("1. Aligning cells between modalities...")
    mu.pp.intersect_obs(mdata)
    
    # Create joint representation by concatenating reduced dimensions
    print("2. Creating joint representation...")
    
    # Use 30 PCs from RNA and 30 LSIs from ATAC
    n_pcs_use = min(30, rna.obsm['X_pca'].shape[1], atac.obsm['X_lsi'].shape[1])
    
    joint_rep = np.concatenate([
        rna.obsm['X_pca'][:, :n_pcs_use], 
        atac.obsm['X_lsi'][:, :n_pcs_use]
    ], axis=1)
    
    mdata.obsm['X_joint'] = joint_rep
    
    # Compute neighbors on joint representation
    print("3. Computing neighbors on joint representation...")
    sc.pp.neighbors(mdata, n_neighbors=15, use_rep='X_joint', metric='cosine', random_state=42)
    
    # UMAP on joint representation
    print("4. Computing UMAP...")
    sc.tl.umap(mdata, random_state=42)
    
    # === COPY QC METRICS TO MDATA.OBS ===
    print("\n=== COPYING QC METRICS ===")
    
    # Copy RNA metrics to mdata.obs
    mdata.obs['rna_n_counts'] = rna.obs['n_counts']
    mdata.obs['rna_n_genes'] = rna.obs['n_genes']
    
    # Copy ATAC metrics to mdata.obs  
    mdata.obs['atac_n_counts'] = atac.obs['n_counts']
    mdata.obs['atac_n_peaks'] = atac.obs['n_peaks']
    
    print(f"mdata.obs columns: {list(mdata.obs.columns)}")
    
    # === CREATE 3 SIMPLE PLOTS ===
    print("\n=== CREATING 3 SIMPLE PLOTS ===")
    
    # 1. RNA ONLY UMAP
    fig = plt.figure(figsize=(8, 6))
    sc.pl.umap(mdata, color=['rna_n_counts'], size=20, show=False, 
              title='RNA: Total Counts', color_map='viridis')
    plt.savefig(f'{base_prefix}_umap_rna_only.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created RNA only plot")
    
    # 2. ATAC ONLY UMAP  
    fig = plt.figure(figsize=(8, 6))
    sc.pl.umap(mdata, color=['atac_n_counts'], size=20, show=False,
              title='ATAC: Total Counts', color_map='plasma')
    plt.savefig(f'{base_prefix}_umap_atac_only.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created ATAC only plot")
    
    # 3. RNA+ATAC OVERLAPPED
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get UMAP coordinates
    umap_coords = mdata.obsm['X_umap']
    
    # Normalize RNA and ATAC counts for coloring
    rna_norm = (mdata.obs['rna_n_counts'] - mdata.obs['rna_n_counts'].min()) / \
               (mdata.obs['rna_n_counts'].max() - mdata.obs['rna_n_counts'].min())
    atac_norm = (mdata.obs['atac_n_counts'] - mdata.obs['atac_n_counts'].min()) / \
                (mdata.obs['atac_n_counts'].max() - mdata.obs['atac_n_counts'].min())
    
    # Create RGB colors: Red=ATAC, Green=RNA
    colors = np.zeros((len(rna_norm), 3))
    colors[:, 0] = atac_norm  # Red channel = ATAC
    colors[:, 1] = rna_norm   # Green channel = RNA
    # Blue channel remains 0
    
    # Plot all cells with combined colors
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
               c=colors, s=10, alpha=0.7)
    
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('RNA+ATAC Overlapped (Red=ATAC, Green=RNA)', fontsize=14, fontweight='bold')
    
    # Add simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='High ATAC'),
        Patch(facecolor='green', alpha=0.7, label='High RNA'),
        Patch(facecolor='yellow', alpha=0.7, label='High Both')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{base_prefix}_umap_rna_atac_overlapped.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Created RNA+ATAC overlapped plot")
    
    # Save integrated data
    print(f"\nSaving integrated data to {args.output}")
    mu.write_h5mu(args.output, mdata)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"✓ Output file: {args.output}")
    print(f"✓ Created 3 plots:")
    print(f"  1. {base_prefix}_umap_rna_only.png")
    print(f"  2. {base_prefix}_umap_atac_only.png")
    print(f"  3. {base_prefix}_umap_rna_atac_overlapped.png")

if __name__ == '__main__':
    main()
