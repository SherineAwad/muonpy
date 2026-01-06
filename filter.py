#!/usr/bin/env python3
import argparse
import muon as mu
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def compute_qc(adata, mito_prefix="mt-"):
    """Compute basic QC metrics"""
    # Counts per cell
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=1)
    # Genes/peaks per cell
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if hasattr(adata.X, "A1") else (adata.X > 0).sum(axis=1)
    # Percent mitochondrial (if RNA)
    if adata.var_names.str.lower().str.startswith(mito_prefix).any():
        mito_genes = [g for g in adata.var_names if g.lower().startswith(mito_prefix)]
        mito_idx = [adata.var_names.get_loc(g) for g in mito_genes]
        adata.obs['percent_mito'] = adata.X[:, mito_idx].sum(axis=1).A1 / adata.obs['n_counts'] * 100
    else:
        adata.obs['percent_mito'] = 0
    return adata

def plot_qc(adata, prefix="before"):
    """Plot QC metrics"""
    df = adata.obs
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    sns.histplot(df['n_genes'], bins=100, kde=True, ax=axes[0])
    axes[0].set_title(f"{prefix}: n_genes per cell")
    sns.histplot(df['n_counts'], bins=100, kde=True, ax=axes[1])
    axes[1].set_title(f"{prefix}: n_counts per cell")
    sns.histplot(df['percent_mito'], bins=50, kde=True, ax=axes[2])
    axes[2].set_title(f"{prefix}: percent_mito per cell")
    plt.tight_layout()
    plt.savefig(f"{prefix}_qc.png")
    plt.close()

def filter_cells(adata, min_genes=200, max_genes=10000, min_counts=500, max_counts=50000, max_mito=50):
    """Apply relaxed filtering thresholds"""
    filtered = adata[
        (adata.obs['n_genes'] >= min_genes) & (adata.obs['n_genes'] <= max_genes) &
        (adata.obs['n_counts'] >= min_counts) & (adata.obs['n_counts'] <= max_counts) &
        (adata.obs['percent_mito'] < max_mito)
    ].copy()
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Filter MuData H5MU cells")
    parser.add_argument("--input", "-i", required=True, help="Input H5MU file")
    parser.add_argument("--output", "-o", default="filtered.h5mu", help="Output H5MU file")
    args = parser.parse_args()

    input_prefix = os.path.splitext(os.path.basename(args.input))[0]

    print(f"Reading {args.input} ...")
    mdata = mu.read_h5mu(args.input)

    # Compute QC for all modalities
    for mod_name, ad in mdata.mod.items():
        print(f"Computing QC for {mod_name} ...")
        mdata.mod[mod_name] = compute_qc(ad)

    # Plot QC before filtering (only RNA for simplicity)
    if 'rna' in mdata.mod:
        print("Plotting QC before filtering (RNA) ...")
        plot_qc(mdata.mod['rna'], prefix=f"{input_prefix}_before")

    # Filter each modality separately (relaxed thresholds)
    filtered_mods = {}
    for mod_name, ad in mdata.mod.items():
        print(f"Filtering {mod_name} ...")
        filtered_mods[mod_name] = filter_cells(ad)

    # Keep union of all barcodes that survived any modality
    all_barcodes = pd.Index([])
    for ad in filtered_mods.values():
        all_barcodes = all_barcodes.union(ad.obs_names)

    print(f"Total cells after filtering (union across modalities): {len(all_barcodes)}")

    # Subset each modality to union barcodes
    for mod_name, ad in mdata.mod.items():
        mdata.mod[mod_name] = ad[ad.obs_names.isin(all_barcodes)].copy()

    # Plot QC after filtering (RNA)
    if 'rna' in mdata.mod:
        print("Plotting QC after filtering (RNA) ...")
        plot_qc(mdata.mod['rna'], prefix=f"{input_prefix}_after")

    print(f"Saving filtered H5MU to {args.output} ...")
    mdata.write(args.output)
    print("Done.")

if __name__ == "__main__":
    main()

