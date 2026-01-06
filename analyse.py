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
    
    # Plot RNA PCA variance
    fig = plt.figure(figsize=(8, 6))
    sc.pl.pca_variance_ratio(rna, log=True, show=False)
    plt.title('RNA PCA variance ratio')
    plt.savefig(f'{base_prefix}_rna_pca_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
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
    
    # Copy PCA to main ATAC object (called X_lsi in tutorials)
    atac.obsm['X_lsi'] = atac_hv.obsm['X_pca']
    atac.uns['lsi'] = atac_hv.uns['pca']
    
    # Plot ATAC PCA variance
    fig = plt.figure(figsize=(8, 6))
    sc.pl.pca_variance_ratio(atac_hv, log=True, show=False)
    plt.title('ATAC LSI variance ratio')
    plt.savefig(f'{base_prefix}_atac_lsi_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
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
    
    # === UMAP VISUALIZATION ===
    print("\n=== CREATING UMAP VISUALIZATIONS ===")
    
    # 1. Basic UMAP (no coloring)
    fig = plt.figure(figsize=(8, 6))
    sc.pl.umap(mdata, size=20, show=False)
    plt.title('UMAP: RNA+ATAC integrated')
    plt.savefig(f'{base_prefix}_umap_integrated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP colored by sample - NEED TO PULL OBS FIRST
    # Pull obs columns from individual modalities to mdata.obs
    mu.pp.intersect_obs(mdata)
    
    # Get column names from each modality
    rna_obs_cols = list(rna.obs.columns)
    atac_obs_cols = list(atac.obs.columns)
    
    print(f"\nRNA obs columns: {rna_obs_cols}")
    print(f"ATAC obs columns: {atac_obs_cols}")
    
    # Create QC plots with actual available columns
    available_columns = list(mdata.obs.columns)
    print(f"\nAvailable columns in mdata.obs: {available_columns}")
    
    # Plot sample if available
    if 'sample' in mdata.obs.columns:
        fig = plt.figure(figsize=(8, 6))
        sc.pl.umap(mdata, color=['sample'], size=20, show=False)
        plt.title('UMAP: Colored by sample')
        plt.savefig(f'{base_prefix}_umap_by_sample.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Plot QC metrics that are actually in mdata.obs
    qc_plots_created = 0
    
    # Check each possible column and plot if available
    possible_qc_columns = [
        ('n_counts', 'Total counts'),
        ('log_n_counts', 'Log total counts'),
        ('n_genes', 'Number of genes'),
        ('n_peaks', 'Number of peaks')
    ]
    
    # Create individual plots for available QC metrics
    for col_name, col_title in possible_qc_columns:
        if col_name in mdata.obs.columns:
            fig = plt.figure(figsize=(8, 6))
            sc.pl.umap(mdata, color=[col_name], size=20, show=False)
            plt.title(f'UMAP: {col_title}')
            plt.savefig(f'{base_prefix}_umap_{col_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            qc_plots_created += 1
            print(f"Created UMAP plot for {col_name}")
    
    # Also check for modality-specific columns (rna_, atac_ prefix)
    for prefix in ['rna_', 'atac_']:
        for col_name, col_title in possible_qc_columns:
            full_col = f"{prefix}{col_name}"
            if full_col in mdata.obs.columns:
                fig = plt.figure(figsize=(8, 6))
                sc.pl.umap(mdata, color=[full_col], size=20, show=False)
                plt.title(f'UMAP: {prefix.upper()}{col_title}')
                plt.savefig(f'{base_prefix}_umap_{full_col}.png', dpi=150, bbox_inches='tight')
                plt.close()
                qc_plots_created += 1
                print(f"Created UMAP plot for {full_col}")
    
    # Create a multi-panel QC plot if we have at least 2 metrics
    if qc_plots_created >= 2:
        # Get the first 4 available QC columns
        qc_cols_to_plot = []
        for col_name, col_title in possible_qc_columns:
            if col_name in mdata.obs.columns and len(qc_cols_to_plot) < 4:
                qc_cols_to_plot.append((col_name, col_title))
        
        # Also check prefixed columns
        if len(qc_cols_to_plot) < 4:
            for prefix in ['rna_', 'atac_']:
                for col_name, col_title in possible_qc_columns:
                    full_col = f"{prefix}{col_name}"
                    if full_col in mdata.obs.columns and len(qc_cols_to_plot) < 4:
                        qc_cols_to_plot.append((full_col, f"{prefix.upper()}{col_title}"))
        
        if len(qc_cols_to_plot) >= 2:
            n_rows = 2
            n_cols = 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, (col_name, col_title) in enumerate(qc_cols_to_plot[:4]):
                sc.pl.umap(mdata, color=[col_name], size=20, show=False, ax=axes[i])
                axes[i].set_title(col_title)
            
            # Hide unused subplots
            for i in range(len(qc_cols_to_plot), 4):
                axes[i].set_visible(False)
            
            plt.savefig(f'{base_prefix}_umap_qc_panel.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Created multi-panel QC UMAP plot")
    
    print(f"\nCreated {qc_plots_created} QC UMAP plots")
    
    # Save integrated data for clustering in separate script
    print(f"\nSaving integrated data to {args.output}")
    mu.write_h5mu(args.output, mdata)
    
    print("\n=== UMAP ANALYSIS COMPLETE ===")
    print(f"  Output file: {args.output}")
    print(f"  Plots saved with prefix: {base_prefix}_")
    print(f"  Total UMAP plots created: {1 + ('sample' in mdata.obs.columns) + qc_plots_created}")

if __name__ == '__main__':
    main()
