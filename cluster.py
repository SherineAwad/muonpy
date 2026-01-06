#!/usr/bin/env python3
"""
cluster.py - Cluster single-cell multiome data (RNA + ATAC)
Usage: python cluster.py -i filtered.h5mu -o clustered.h5mu
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
    parser = argparse.ArgumentParser(description='Cluster single-cell multiome data')
    parser.add_argument('-i', '--input', required=True, help='Input h5mu file')
    parser.add_argument('-o', '--output', required=True, help='Output h5mu file')
    parser.add_argument('--rna-resolution', type=float, default=0.8, help='Resolution for RNA clustering (default: 0.8)')
    parser.add_argument('--atac-resolution', type=float, default=0.8, help='Resolution for ATAC clustering (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    return parser.parse_args()

def process_rna(rna, resolution, seed):
    """Process RNA data for clustering"""
    print("\n=== PROCESSING RNA DATA ===")
    print(f"RNA shape: {rna.shape}")
    
    # Normalize and log transform
    print("Normalizing and log transforming...")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    
    # Find highly variable genes
    print("Finding highly variable genes...")
    sc.pp.highly_variable_genes(rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"Found {rna.var['highly_variable'].sum()} highly variable genes")
    
    # Scale data
    print("Scaling data...")
    sc.pp.scale(rna, max_value=10)
    
    # PCA
    print("Running PCA...")
    sc.tl.pca(rna, svd_solver='arpack', random_state=seed)
    
    # Compute neighbors
    print("Computing neighbors graph...")
    sc.pp.neighbors(rna, n_neighbors=15, n_pcs=30, random_state=seed)
    
    # UMAP
    print("Computing UMAP...")
    sc.tl.umap(rna, random_state=seed)
    
    # Leiden clustering
    print(f"Clustering with resolution {resolution}...")
    sc.tl.leiden(rna, resolution=resolution, random_state=seed, key_added='rna_cluster')
    
    # Calculate cluster sizes
    cluster_counts = rna.obs['rna_cluster'].value_counts().sort_index()
    print(f"RNA clusters: {len(cluster_counts)}")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells ({count/rna.n_obs*100:.1f}%)")
    
    return rna

def process_atac(atac, resolution, seed):
    """Process ATAC data for clustering"""
    print("\n=== PROCESSING ATAC DATA ===")
    print(f"ATAC shape: {atac.shape}")
    
    # Binarize the data (ATAC is binary)
    print("Binarizing ATAC data...")
    atac.X.data[:] = 1
    
    # TF-IDF normalization
    print("Applying TF-IDF normalization...")
    n_cells = atac.shape[0]
    n_features = atac.shape[1]
    
    # Term frequency
    tf = atac.X / atac.X.sum(axis=1)
    
    # Inverse document frequency
    n_cells_with_feature = np.array((atac.X > 0).sum(axis=0)).flatten()
    idf = np.log(n_cells / n_cells_with_feature) + 1
    
    # TF-IDF
    from scipy import sparse
    if hasattr(tf, 'A'):
        tf_idf = tf.A * idf
        atac.X = sparse.csr_matrix(tf_idf)
    else:
        tf_idf = tf.multiply(idf)
        atac.X = tf_idf
    
    # Latent Semantic Indexing (LSI) - similar to PCA for ATAC
    print("Running LSI (ATAC PCA)...")
    sc.tl.pca(atac, svd_solver='arpack', random_state=seed)
    
    # Compute neighbors
    print("Computing neighbors graph...")
    sc.pp.neighbors(atac, n_neighbors=15, n_pcs=30, random_state=seed)
    
    # UMAP
    print("Computing UMAP...")
    sc.tl.umap(atac, random_state=seed)
    
    # Leiden clustering
    print(f"Clustering with resolution {resolution}...")
    sc.tl.leiden(atac, resolution=resolution, random_state=seed, key_added='atac_cluster')
    
    # Calculate cluster sizes
    cluster_counts = atac.obs['atac_cluster'].value_counts().sort_index()
    print(f"ATAC clusters: {len(cluster_counts)}")
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} cells ({count/atac.n_obs*100:.1f}%)")
    
    return atac

def create_cluster_plots(rna, atac, mdata, prefix):
    """Create UMAP plots with cluster labels"""
    
    # RNA UMAP plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get unique clusters and assign colors
    rna_clusters = sorted(rna.obs['rna_cluster'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(rna_clusters)))
    
    # Plot each cluster with its label
    for i, cluster in enumerate(rna_clusters):
        mask = rna.obs['rna_cluster'] == cluster
        ax.scatter(rna.obsm['X_umap'][mask, 0], 
                  rna.obsm['X_umap'][mask, 1], 
                  c=[colors[i]], s=5, alpha=0.7, label=f'{cluster}')
        
        # Calculate centroid for label position
        centroid_x = np.mean(rna.obsm['X_umap'][mask, 0])
        centroid_y = np.mean(rna.obsm['X_umap'][mask, 1])
        
        # Add cluster label
        ax.text(centroid_x, centroid_y, str(cluster), 
               fontsize=12, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'RNA Clusters (n={len(rna_clusters)})')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_rna_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ATAC UMAP plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get unique clusters and assign colors
    atac_clusters = sorted(atac.obs['atac_cluster'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(atac_clusters)))
    
    # Plot each cluster with its label
    for i, cluster in enumerate(atac_clusters):
        mask = atac.obs['atac_cluster'] == cluster
        ax.scatter(atac.obsm['X_umap'][mask, 0], 
                  atac.obsm['X_umap'][mask, 1], 
                  c=[colors[i]], s=5, alpha=0.7, label=f'{cluster}')
        
        # Calculate centroid for label position
        centroid_x = np.mean(atac.obsm['X_umap'][mask, 0])
        centroid_y = np.mean(atac.obsm['X_umap'][mask, 1])
        
        # Add cluster label
        ax.text(centroid_x, centroid_y, str(cluster), 
               fontsize=12, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'ATAC Clusters (n={len(atac_clusters)})')
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f'{prefix}_atac_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Combined multiome UMAP (RNA)
    if 'rna' in mdata.mod and 'atac' in mdata.mod:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # RNA clusters
        rna_clusters = sorted(rna.obs['rna_cluster'].unique())
        colors_rna = plt.cm.tab20(np.linspace(0, 1, len(rna_clusters)))
        
        for i, cluster in enumerate(rna_clusters):
            mask = rna.obs['rna_cluster'] == cluster
            ax1.scatter(rna.obsm['X_umap'][mask, 0], 
                       rna.obsm['X_umap'][mask, 1], 
                       c=[colors_rna[i]], s=5, alpha=0.7)
            
            centroid_x = np.mean(rna.obsm['X_umap'][mask, 0])
            centroid_y = np.mean(rna.obsm['X_umap'][mask, 1])
            ax1.text(centroid_x, centroid_y, str(cluster), 
                    fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.set_title(f'RNA Clusters (n={len(rna_clusters)})')
        ax1.grid(False)
        
        # ATAC clusters
        atac_clusters = sorted(atac.obs['atac_cluster'].unique())
        colors_atac = plt.cm.tab20(np.linspace(0, 1, len(atac_clusters)))
        
        for i, cluster in enumerate(atac_clusters):
            mask = atac.obs['atac_cluster'] == cluster
            ax2.scatter(atac.obsm['X_umap'][mask, 0], 
                       atac.obsm['X_umap'][mask, 1], 
                       c=[colors_atac[i]], s=5, alpha=0.7)
            
            centroid_x = np.mean(atac.obsm['X_umap'][mask, 0])
            centroid_y = np.mean(atac.obsm['X_umap'][mask, 1])
            ax2.text(centroid_x, centroid_y, str(cluster), 
                    fontsize=10, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        ax2.set_title(f'ATAC Clusters (n={len(atac_clusters)})')
        ax2.grid(False)
        
        plt.tight_layout()
        plt.savefig(f'{prefix}_combined_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    args = parse_args()
    
    base_prefix = os.path.splitext(os.path.basename(args.input))[0]
    
    print(f"Loading data from {args.input}")
    mdata = mu.read_h5mu(args.input)
    
    print(f"Available modalities: {list(mdata.mod.keys())}")
    print(f"Random seed: {args.seed}")
    print(f"RNA resolution: {args.rna_resolution}")
    print(f"ATAC resolution: {args.atac_resolution}")
    
    np.random.seed(args.seed)
    
    # Process RNA if available
    if 'rna' in mdata.mod:
        rna = mdata.mod['rna'].copy()
        rna = process_rna(rna, args.rna_resolution, args.seed)
        mdata.mod['rna'] = rna
    
    # Process ATAC if available
    if 'atac' in mdata.mod:
        atac = mdata.mod['atac'].copy()
        atac = process_atac(atac, args.atac_resolution, args.seed)
        mdata.mod['atac'] = atac
    
    # Create cluster plots
    print("\n=== CREATING CLUSTER PLOTS ===")
    create_cluster_plots(rna if 'rna' in mdata.mod else None,
                        atac if 'atac' in mdata.mod else None,
                        mdata, base_prefix)
    
    # Save clustered data
    print(f"\nSaving clustered data to {args.output}")
    mu.write_h5mu(args.output, mdata)
    
    print("\n=== CLUSTERING COMPLETE ===")
    print(f"  Output file: {args.output}")
    print(f"  Cluster plots:")
    if 'rna' in mdata.mod:
        print(f"    {base_prefix}_rna_clusters.png")
    if 'atac' in mdata.mod:
        print(f"    {base_prefix}_atac_clusters.png")
    if 'rna' in mdata.mod and 'atac' in mdata.mod:
        print(f"    {base_prefix}_combined_clusters.png")
    
    # Print summary
    print(f"\n=== CLUSTER SUMMARY ===")
    if 'rna' in mdata.mod:
        rna_clusters = len(mdata.mod['rna'].obs['rna_cluster'].unique())
        print(f"RNA: {rna_clusters} clusters")
    if 'atac' in mdata.mod:
        atac_clusters = len(mdata.mod['atac'].obs['atac_cluster'].unique())
        print(f"ATAC: {atac_clusters} clusters")

if __name__ == '__main__':
    main()
