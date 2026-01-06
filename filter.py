#!/usr/bin/env python3
"""
filter.py - Filter single-cell multiome data (RNA + ATAC) with proper QC
Usage: python filter.py -i input.h5mu -o filtered.h5mu
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

def parse_args():
    parser = argparse.ArgumentParser(description='Filter single-cell multiome data (RNA + ATAC)')
    parser.add_argument('-i', '--input', required=True, help='Input h5mu file')
    parser.add_argument('-o', '--output', required=True, help='Output h5mu file')
    return parser.parse_args()

def create_rna_plots(rna, prefix, title_suffix=""):
    """Create RNA QC plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'RNA QC {title_suffix}', fontsize=16, y=1.02)
    
    # 1. Histogram of nCount_RNA
    axes[0, 0].hist(rna.obs['n_counts'], bins=50, edgecolor='black')
    axes[0, 0].axvline(x=1000, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(x=50000, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('nCount_RNA')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title('Total RNA counts per cell')
    
    # 2. Histogram of nFeature_RNA
    axes[0, 1].hist(rna.obs['n_genes'], bins=50, edgecolor='black')
    axes[0, 1].axvline(x=200, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axvline(x=5000, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('nFeature_RNA')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].set_title('Number of genes per cell')
    
    # 3. Scatter: nFeature_RNA vs nCount_RNA colored by percent_mt
    if 'percent_mt' in rna.obs:
        scatter = axes[1, 0].scatter(rna.obs['n_counts'], rna.obs['n_genes'], 
                                    c=rna.obs['percent_mt'], s=5, alpha=0.7, 
                                    cmap='viridis', vmax=30)
        axes[1, 0].set_xlabel('nCount_RNA')
        axes[1, 0].set_ylabel('nFeature_RNA')
        axes[1, 0].set_title('Genes vs counts (colored by %MT)')
        plt.colorbar(scatter, ax=axes[1, 0], label='% Mitochondrial')
    else:
        axes[1, 0].scatter(rna.obs['n_counts'], rna.obs['n_genes'], s=5, alpha=0.7)
        axes[1, 0].set_xlabel('nCount_RNA')
        axes[1, 0].set_ylabel('nFeature_RNA')
        axes[1, 0].set_title('Genes vs counts')
    
    # 4. Histogram of percent.mt
    if 'percent_mt' in rna.obs:
        axes[1, 1].hist(rna.obs['percent_mt'], bins=50, edgecolor='black')
        axes[1, 1].axvline(x=20, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Percent mitochondrial')
        axes[1, 1].set_ylabel('Number of cells')
        axes[1, 1].set_title('Mitochondrial percentage')
    else:
        axes[1, 1].hist(rna.obs['n_counts'], bins=50, edgecolor='black')
        axes[1, 1].axvline(x=1000, color='red', linestyle='--', linewidth=2)
        axes[1, 1].axvline(x=50000, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('nCount_RNA')
        axes[1, 1].set_ylabel('Number of cells')
        axes[1, 1].set_title('Total RNA counts (alternative)')
    
    plt.tight_layout()
    plt.savefig(f'{prefix}_rna_qc{title_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_atac_plots(atac, prefix, title_suffix=""):
    """Create ATAC QC plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'ATAC QC {title_suffix}', fontsize=16, y=1.02)
    
    # 1. Histogram of nCount_ATAC
    axes[0, 0].hist(atac.obs['n_counts'], bins=50, edgecolor='black')
    axes[0, 0].axvline(x=1000, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(x=100000, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('nCount_ATAC')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title('Total ATAC counts per cell')
    
    # 2. Histogram of nFeature_ATAC
    axes[0, 1].hist(atac.obs['n_peaks'], bins=50, edgecolor='black')
    axes[0, 1].axvline(x=500, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('nFeature_ATAC')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].set_title('Number of peaks per cell')
    
    # 3. Scatter: nFeature_ATAC vs nCount_ATAC
    if 'tss_enrichment' in atac.obs:
        scatter = axes[1, 0].scatter(atac.obs['n_counts'], atac.obs['n_peaks'], 
                                    c=atac.obs['tss_enrichment'], s=5, alpha=0.7, 
                                    cmap='plasma', vmax=0.2)
        axes[1, 0].set_xlabel('nCount_ATAC')
        axes[1, 0].set_ylabel('nFeature_ATAC')
        axes[1, 0].set_title('Peaks vs counts (colored by TSS enrichment)')
        plt.colorbar(scatter, ax=axes[1, 0], label='TSS enrichment')
    else:
        axes[1, 0].scatter(atac.obs['n_counts'], atac.obs['n_peaks'], s=5, alpha=0.7)
        axes[1, 0].set_xlabel('nCount_ATAC')
        axes[1, 0].set_ylabel('nFeature_ATAC')
        axes[1, 0].set_title('Peaks vs counts')
    
    # 4. Histogram of TSS enrichment
    if 'tss_enrichment' in atac.obs:
        axes[1, 1].hist(atac.obs['tss_enrichment'], bins=50, edgecolor='black')
        axes[1, 1].axvline(x=0.1, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('TSS enrichment')
        axes[1, 1].set_ylabel('Number of cells')
        axes[1, 1].set_title('TSS enrichment score')
    else:
        axes[1, 1].hist(atac.obs['n_peaks'], bins=50, edgecolor='black')
        axes[1, 1].axvline(x=500, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('nFeature_ATAC')
        axes[1, 1].set_ylabel('Number of cells')
        axes[1, 1].set_title('Number of peaks per cell (alternative)')
    
    plt.tight_layout()
    plt.savefig(f'{prefix}_atac_qc{title_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    base_prefix = os.path.splitext(os.path.basename(args.input))[0]
    
    print(f"Loading data from {args.input}")
    mdata = mu.read_h5mu(args.input)
    
    print(f"Available modalities: {list(mdata.mod.keys())}")
    
    # Store original cell counts
    original_counts = {}
    for mod in mdata.mod.keys():
        original_counts[mod] = mdata.mod[mod].n_obs
    
    # === PROCESS RNA DATA ===
    if 'rna' in mdata.mod:
        print("\n=== PROCESSING RNA DATA ===")
        rna = mdata.mod['rna'].copy()
        print(f"RNA original shape: {rna.shape}")
        
        # Calculate basic RNA QC metrics
        rna.obs['n_counts'] = rna.X.sum(axis=1).A1 if hasattr(rna.X, 'A1') else rna.X.sum(axis=1)
        rna.obs['n_genes'] = (rna.X > 0).sum(axis=1).A1 if hasattr(rna.X, 'A1') else (rna.X > 0).sum(axis=1)
        
        # Calculate mitochondrial percentage
        mt_genes = pd.Series(False, index=rna.var.index)
        
        # Check var_names directly since that's where they are
        var_names_str = pd.Series(rna.var_names.astype(str))
        mt_genes = var_names_str.str.contains('^mt-', case=False, na=False)
        
        if mt_genes.sum() > 0:
            mt_gene_names = list(rna.var_names[mt_genes])
            print(f"Found {mt_genes.sum()} MT genes in var_names")
            print(f"MT gene examples: {mt_gene_names[:10]}")
            
            # Extract MT counts - FIXED: ensure it's 1D array
            mt_data = rna[:, mt_genes].X
            if hasattr(mt_data, 'A1'):
                mt_counts = mt_data.sum(axis=1).A1
            else:
                mt_counts = mt_data.sum(axis=1)
                if hasattr(mt_counts, 'A1'):
                    mt_counts = mt_counts.A1
                elif hasattr(mt_counts, 'toarray'):
                    mt_counts = mt_counts.toarray().flatten()
                else:
                    mt_counts = np.array(mt_counts).flatten()
            
            # Calculate percentage
            rna.obs['percent_mt'] = 100 * mt_counts / (rna.obs['n_counts'].values + 1e-6)
            print(f"MT percentage range: {rna.obs['percent_mt'].min():.2f}% - {rna.obs['percent_mt'].max():.2f}%")
        else:
            rna.obs['percent_mt'] = 0
            print("Warning: No mitochondrial genes found")
        
        # Create BEFORE filtering plots
        create_rna_plots(rna, base_prefix, "_before_filtering")
        
        # === RNA FILTERING ===
        print(f"\nRNA cells before filtering: {rna.n_obs}")
        
        # Apply filters
        mask = (
            (rna.obs['n_counts'] >= 1000) & 
            (rna.obs['n_counts'] <= 50000) &
            (rna.obs['n_genes'] >= 200) &
            (rna.obs['n_genes'] <= 5000) &
            (rna.obs['percent_mt'] <= 20)
        )
        
        rna_filtered = rna[mask, :].copy()
        print(f"RNA cells after filtering: {rna_filtered.n_obs}")
        
        # Filter genes
        print(f"RNA genes before filtering: {rna_filtered.n_vars}")
        sc.pp.filter_genes(rna_filtered, min_cells=10)
        print(f"RNA genes after min_cells=10 filter: {rna_filtered.n_vars}")
        
        # Create AFTER filtering plots
        create_rna_plots(rna_filtered, base_prefix, "_after_filtering")
        
        mdata.mod['rna'] = rna_filtered
    
    # === PROCESS ATAC DATA ===
    if 'atac' in mdata.mod:
        print("\n=== PROCESSING ATAC DATA ===")
        atac = mdata.mod['atac'].copy()
        print(f"ATAC original shape: {atac.shape}")
        
        # Calculate basic ATAC QC metrics
        atac.obs['n_counts'] = atac.X.sum(axis=1).A1 if hasattr(atac.X, 'A1') else atac.X.sum(axis=1)
        atac.obs['n_peaks'] = (atac.X > 0).sum(axis=1).A1 if hasattr(atac.X, 'A1') else (atac.X > 0).sum(axis=1)
        
        # Calculate TSS enrichment if possible
        atac.obs['tss_enrichment'] = 0  # Default
        
        # Look for TSS annotation in var columns
        for col in atac.var.columns:
            if atac.var[col].dtype == object:
                col_data = atac.var[col].astype(str)
                tss_mask = col_data.str.contains('TSS|promoter|start', case=False, na=False)
                if tss_mask.sum() > 0:
                    tss_data = atac[:, tss_mask].X
                    if hasattr(tss_data, 'A1'):
                        tss_counts = tss_data.sum(axis=1).A1
                    else:
                        tss_counts = tss_data.sum(axis=1)
                        if hasattr(tss_counts, 'A1'):
                            tss_counts = tss_counts.A1
                        elif hasattr(tss_counts, 'toarray'):
                            tss_counts = tss_counts.toarray().flatten()
                        else:
                            tss_counts = np.array(tss_counts).flatten()
                    
                    total_counts = atac.obs['n_counts'].values
                    atac.obs['tss_enrichment'] = tss_counts / (total_counts + 1e-6)
                    print(f"Found {tss_mask.sum()} TSS peaks in column '{col}'")
                    if atac.obs['tss_enrichment'].max() > 0:
                        print(f"TSS enrichment range: {atac.obs['tss_enrichment'].min():.4f} - {atac.obs['tss_enrichment'].max():.4f}")
                    break
        
        # Create BEFORE filtering plots
        create_atac_plots(atac, base_prefix, "_before_filtering")
        
        # === ATAC FILTERING ===
        print(f"\nATAC cells before filtering: {atac.n_obs}")
        
        # Apply filters
        mask = (
            (atac.obs['n_counts'] >= 1000) & 
            (atac.obs['n_counts'] <= 100000) &
            (atac.obs['n_peaks'] >= 500)
        )
        
        if atac.obs['tss_enrichment'].max() > 0:
            mask = mask & (atac.obs['tss_enrichment'] >= 0.1)
            print("Applied TSS enrichment filter (≥0.1)")
        
        atac_filtered = atac[mask, :].copy()
        print(f"ATAC cells after filtering: {atac_filtered.n_obs}")
        
        # Filter peaks
        print(f"ATAC peaks before filtering: {atac_filtered.n_vars}")
        sc.pp.filter_genes(atac_filtered, min_cells=10)
        print(f"ATAC peaks after min_cells=10 filter: {atac_filtered.n_vars}")
        
        # Remove non-standard chromosomes if available
        if 'chrom' in atac_filtered.var.columns:
            print("\nRemoving peaks on non-standard chromosomes...")
            chromosomes = atac_filtered.var['chrom'].astype(str)
            valid_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'M', 'MT']
            valid_chromosomes += [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
            chr_mask = chromosomes.isin(valid_chromosomes)
            atac_filtered = atac_filtered[:, chr_mask].copy()
            print(f"ATAC peaks after chromosome filter: {atac_filtered.n_vars}")
        
        # Create AFTER filtering plots
        create_atac_plots(atac_filtered, base_prefix, "_after_filtering")
        
        mdata.mod['atac'] = atac_filtered
    
    # === ALIGN CELLS ===
    print("\n=== ALIGNING CELLS BETWEEN MODALITIES ===")
    mu.pp.intersect_obs(mdata)
    
    print(f"\n=== SUMMARY ===")
    for mod in original_counts.keys():
        if mod in mdata.mod:
            print(f"{mod.upper()}:")
            print(f"  Before filtering: {original_counts[mod]} cells")
            print(f"  After filtering:  {mdata.mod[mod].n_obs} cells")
            if mod in original_counts:
                percent_kept = (mdata.mod[mod].n_obs / original_counts[mod]) * 100
                print(f"  Kept: {percent_kept:.1f}%")
    
    # Save data
    print(f"\nSaving filtered data to {args.output}")
    mu.write_h5mu(args.output, mdata)
    
    print("\n=== FILTERING COMPLETE ===")
    print(f"  Output file: {args.output}")
    print(f"  QC plots:")
    for suffix in ["_before_filtering", "_after_filtering"]:
        print(f"    {base_prefix}_rna_qc{suffix}.png")
        print(f"    {base_prefix}_atac_qc{suffix}.png")

if __name__ == '__main__':
    main()
