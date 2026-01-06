#!/usr/bin/env python3
"""
filter.py - Filter single-cell ATAC data from PBMC tutorial
Usage: python filter.py -i input.h5mu -o filtered.h5mu
"""

import argparse
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Filter single-cell ATAC data')
    parser.add_argument('-i', '--input', required=True, help='Input h5mu file')
    parser.add_argument('-o', '--output', required=True, help='Output h5mu file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get base prefix for output files
    base_prefix = os.path.splitext(os.path.basename(args.input))[0]
    
    print(f"Loading data from {args.input}")
    mdata = mu.read_h5mu(args.input)
    atac = mdata.mod['atac']
    
    print("Original shape:", atac.shape)
    
    # Store initial metrics - handle division by zero in log transform
    atac.obs['n_counts'] = atac.X.sum(axis=1).A1
    # Add small epsilon to avoid log10(0)
    atac.obs['log_n_counts'] = np.log10(atac.obs['n_counts'].values + 1)
    atac.obs['n_peaks'] = (atac.X > 0).sum(axis=1).A1
    atac.var['n_cells'] = (atac.X > 0).sum(axis=0).A1
    # Add small epsilon to avoid log10(0)
    atac.var['log_n_cells'] = np.log10(atac.var['n_cells'].values + 1)
    
    # Plot QC metrics before filtering
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total counts per cell
    axes[0, 0].hist(atac.obs['n_counts'], bins=50)
    axes[0, 0].set_xlabel('Total counts')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].axvline(x=1000, color='r', linestyle='--', label='Min counts=1000')
    axes[0, 0].legend()
    axes[0, 0].set_title('Total counts per cell')
    
    # Peaks per cell
    axes[0, 1].hist(atac.obs['n_peaks'], bins=50)
    axes[0, 1].set_xlabel('Number of peaks')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].axvline(x=500, color='r', linestyle='--', label='Min peaks=500')
    axes[0, 1].legend()
    axes[0, 1].set_title('Peaks per cell')
    
    # Cells per peak
    axes[1, 0].hist(atac.var['n_cells'], bins=50)
    axes[1, 0].set_xlabel('Number of cells')
    axes[1, 0].set_ylabel('Number of peaks')
    axes[1, 0].axvline(x=10, color='r', linestyle='--', label='Min cells=10')
    axes[1, 0].legend()
    axes[1, 0].set_title('Cells per peak')
    
    # Scatter: peaks vs counts
    axes[1, 1].scatter(atac.obs['n_counts'], atac.obs['n_peaks'], alpha=0.5, s=5)
    axes[1, 1].set_xlabel('Total counts')
    axes[1, 1].set_ylabel('Number of peaks')
    axes[1, 1].set_title('Peaks vs total counts')
    
    plt.tight_layout()
    plt.savefig(f'{base_prefix}_qc_before_filtering.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Filter cells based on counts and peaks
    print("\nFiltering cells...")
    print(f"Cells before filtering: {atac.n_obs}")
    
    # Basic filtering
    sc.pp.filter_cells(atac, min_counts=1000)
    print(f"Cells after min_counts=1000 filter: {atac.n_obs}")
    
    sc.pp.filter_cells(atac, min_genes=500)  # min_genes is used for min_peaks in ATAC
    print(f"Cells after min_genes=500 filter: {atac.n_obs}")
    
    # Filter peaks
    print("\nFiltering peaks...")
    print(f"Peaks before filtering: {atac.n_vars}")
    
    sc.pp.filter_genes(atac, min_cells=10)  # min_cells filter for peaks
    print(f"Peaks after min_cells=10 filter: {atac.n_vars}")
    
    # Check if chrom column exists before trying to filter by chromosome
    if 'chrom' in atac.var.columns:
        print("\nRemoving peaks on non-standard chromosomes...")
        chromosomes = atac.var['chrom'].astype(str)
        # Keep standard chromosomes (1-22, X, Y, M/MT)
        valid_chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'M', 'MT', 'chrM']
        # Also handle 'chr' prefix if present
        valid_chromosomes += [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']
        mask = chromosomes.isin(valid_chromosomes)
        atac = atac[:, mask].copy()
        print(f"Peaks after chromosome filter: {atac.n_vars}")
    else:
        print("\nWarning: 'chrom' column not found in var dataframe. Skipping chromosome filtering.")
        print("Available var columns:", list(atac.var.columns))
    
    # Update metrics after filtering
    atac.obs['n_counts'] = atac.X.sum(axis=1).A1
    atac.obs['log_n_counts'] = np.log10(atac.obs['n_counts'].values + 1)
    atac.obs['n_peaks'] = (atac.X > 0).sum(axis=1).A1
    atac.var['n_cells'] = (atac.X > 0).sum(axis=0).A1
    atac.var['log_n_cells'] = np.log10(atac.var['n_cells'].values + 1)
    
    # Plot QC metrics after filtering
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(atac.obs['n_counts'], bins=50)
    axes[0, 0].set_xlabel('Total counts')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title('Total counts per cell (filtered)')
    
    axes[0, 1].hist(atac.obs['n_peaks'], bins=50)
    axes[0, 1].set_xlabel('Number of peaks')
    axes[0, 1].set_ylabel('Number of cells')
    axes[0, 1].set_title('Peaks per cell (filtered)')
    
    axes[1, 0].hist(atac.var['n_cells'], bins=50)
    axes[1, 0].set_xlabel('Number of cells')
    axes[1, 0].set_ylabel('Number of peaks')
    axes[1, 0].set_title('Cells per peak (filtered)')
    
    axes[1, 1].scatter(atac.obs['n_counts'], atac.obs['n_peaks'], alpha=0.5, s=5)
    axes[1, 1].set_xlabel('Total counts')
    axes[1, 1].set_ylabel('Number of peaks')
    axes[1, 1].set_title('Peaks vs total counts (filtered)')
    
    plt.tight_layout()
    plt.savefig(f'{base_prefix}_qc_after_filtering.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Update the mdata object
    mdata.mod['atac'] = atac
    
    # Save the filtered data
    print(f"\nSaving filtered data to {args.output}")
    mu.write_h5mu(args.output, mdata)
    
    print("\nFiltering summary:")
    print(f"  Final cell count: {atac.n_obs}")
    print(f"  Final peak count: {atac.n_vars}")
    print(f"  QC plots saved as: {base_prefix}_qc_before_filtering.png")
    print(f"                    {base_prefix}_qc_after_filtering.png")

if __name__ == '__main__':
    main()
