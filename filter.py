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

def main():
    args = parse_args()
    
    base_prefix = os.path.splitext(os.path.basename(args.input))[0]
    
    print(f"Loading data from {args.input}")
    mdata = mu.read_h5mu(args.input)
    
    print(f"Available modalities: {list(mdata.mod.keys())}")
    
    # === PROCESS RNA DATA ===
    if 'rna' in mdata.mod:
        print("\n=== PROCESSING RNA DATA ===")
        rna = mdata.mod['rna']
        print(f"RNA original shape: {rna.shape}")
        
        # Calculate basic RNA QC metrics
        rna.obs['n_counts'] = rna.X.sum(axis=1).A1 if hasattr(rna.X, 'A1') else rna.X.sum(axis=1)
        rna.obs['n_genes'] = (rna.X > 0).sum(axis=1).A1 if hasattr(rna.X, 'A1') else (rna.X > 0).sum(axis=1)
        
        # Calculate mitochondrial percentage
        mt_genes = pd.Series(False, index=rna.var.index)
        for col in rna.var.columns:
            if rna.var[col].dtype == object:
                col_data = rna.var[col].astype(str)
                mt_genes = mt_genes | col_data.str.contains('^MT-|^mt-|^Mt-', na=False)
        
        if mt_genes.sum() > 0:
            mt_counts = rna[:, mt_genes].X.sum(axis=1).A1 if hasattr(rna[:, mt_genes].X, 'A1') else rna[:, mt_genes].X.sum(axis=1)
            rna.obs['percent_mt'] = 100 * mt_counts / (rna.obs['n_counts'].values + 1e-6)
            print(f"Found {mt_genes.sum()} mitochondrial genes")
        else:
            rna.obs['percent_mt'] = 0
            print("Warning: No mitochondrial genes found")
        
        # === RNA PLOTS (EXACTLY LIKE TUTORIAL) ===
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
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
        
        # 3. Scatter: nFeature_RNA vs nCount_RNA colored by percent.mt
        scatter = axes[1, 0].scatter(rna.obs['n_counts'], rna.obs['n_genes'], 
                                    c=rna.obs['percent_mt'], s=5, alpha=0.7, 
                                    cmap='viridis', vmax=30)
        axes[1, 0].set_xlabel('nCount_RNA')
        axes[1, 0].set_ylabel('nFeature_RNA')
        axes[1, 0].set_title('Genes vs counts (colored by %MT)')
        plt.colorbar(scatter, ax=axes[1, 0], label='% Mitochondrial')
        
        # 4. Histogram of percent.mt
        axes[1, 1].hist(rna.obs['percent_mt'], bins=50, edgecolor='black')
        axes[1, 1].axvline(x=20, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Percent mitochondrial')
        axes[1, 1].set_ylabel('Number of cells')
        axes[1, 1].set_title('Mitochondrial percentage')
        
        plt.tight_layout()
        plt.savefig(f'{base_prefix}_rna_qc.png', dpi=150, bbox_inches='tight')
        plt.close()
        
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
        
        rna = rna[mask, :].copy()
        print(f"RNA cells after filtering: {rna.n_obs}")
        
        # Filter genes
        print(f"RNA genes before filtering: {rna.n_vars}")
        sc.pp.filter_genes(rna, min_cells=10)
        print(f"RNA genes after min_cells=10 filter: {rna.n_vars}")
        
        mdata.mod['rna'] = rna
    
    # === PROCESS ATAC DATA ===
    if 'atac' in mdata.mod:
        print("\n=== PROCESSING ATAC DATA ===")
        atac = mdata.mod['atac']
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
                    tss_counts = atac[:, tss_mask].X.sum(axis=1).A1 if hasattr(atac[:, tss_mask].X, 'A1') else atac[:, tss_mask].X.sum(axis=1)
                    total_counts = atac.obs['n_counts'].values
                    atac.obs['tss_enrichment'] = tss_counts / (total_counts + 1e-6)
                    print(f"Found {tss_mask.sum()} TSS peaks in column '{col}'")
                    break
        
        # === ATAC PLOTS (EXACTLY LIKE TUTORIAL) ===
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
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
        
        # 3. Scatter: nFeature_ATAC vs nCount_ATAC colored by TSS enrichment
        if atac.obs['tss_enrichment'].max() > 0:
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
            axes[1, 0].set_title('Peaks vs counts (TSS data not available)')
        
        # 4. Histogram of TSS enrichment
        if atac.obs['tss_enrichment'].max() > 0:
            axes[1, 1].hist(atac.obs['tss_enrichment'], bins=50, edgecolor='black')
            axes[1, 1].axvline(x=0.1, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('TSS enrichment')
            axes[1, 1].set_ylabel('Number of cells')
            axes[1, 1].set_title('TSS enrichment score')
        else:
            axes[1, 1].text(0.5, 0.5, 'TSS annotation\nnot available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes,
                          fontsize=12)
            axes[1, 1].set_title('TSS enrichment (data not available)')
        
        plt.tight_layout()
        plt.savefig(f'{base_prefix}_atac_qc.png', dpi=150, bbox_inches='tight')
        plt.close()
        
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
        
        atac = atac[mask, :].copy()
        print(f"ATAC cells after filtering: {atac.n_obs}")
        
        # Filter peaks
        print(f"ATAC peaks before filtering: {atac.n_vars}")
        sc.pp.filter_genes(atac, min_cells=10)
        print(f"ATAC peaks after min_cells=10 filter: {atac.n_vars}")
        
        # Remove non-standard chromosomes if available
        if 'chrom' in atac.var.columns:
            print("\nRemoving peaks on non-standard chromosomes...")
            chromosomes = atac.var['chrom'].astype(str)
            valid_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'M', 'MT']
            valid_chromosomes += [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
            chr_mask = chromosomes.isin(valid_chromosomes)
            atac = atac[:, chr_mask].copy()
            print(f"ATAC peaks after chromosome filter: {atac.n_vars}")
        
        mdata.mod['atac'] = atac
    
    # === ALIGN CELLS ===
    print("\n=== ALIGNING CELLS BETWEEN MODALITIES ===")
    mu.pp.intersect_obs(mdata)
    
    print(f"\nFinal cell counts after alignment:")
    for mod in mdata.mod.keys():
        print(f"  {mod.upper()}: {mdata.mod[mod].n_obs} cells, {mdata.mod[mod].n_vars} features")
    
    # Save data
    print(f"\nSaving filtered data to {args.output}")
    mu.write_h5mu(args.output, mdata)
    
    print("\n=== FILTERING COMPLETE ===")
    print(f"  Output file: {args.output}")
    print(f"  QC plots: {base_prefix}_rna_qc.png, {base_prefix}_atac_qc.png")

if __name__ == '__main__':
    main()
