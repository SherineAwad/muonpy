#!/usr/bin/env python3
import argparse
import pandas as pd
import muon as mu
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread

def build_mdata(sample):
    print(f"\nProcessing sample: {sample}")

    # --------------------------
    # Load ATAC
    # --------------------------
    peaks_file = f"{sample}_peaks.tsv"
    barcodes_file = f"{sample}_barcodes.tsv"
    counts_file = f"{sample}_atac_counts.mtx"

    peak_ids = pd.read_csv(peaks_file, header=None)[0].values
    barcodes = pd.read_csv(barcodes_file, header=None)[0].values
    peak_cell_matrix = mmread(counts_file).tocsr()

    atac = AnnData(X=peak_cell_matrix.T)  # transpose: rows=cells, cols=peaks
    atac.obs['barcode'] = barcodes
    atac.var['peak'] = peak_ids
    atac.var_names_make_unique()

    # --------------------------
    # Load RNA
    # --------------------------
    rna_h5 = f"{sample}_filtered_feature_bc_matrix.h5"
    rna = sc.read_10x_h5(rna_h5)
    rna.var_names_make_unique()

    # --------------------------
    # Combine into MuData
    # --------------------------
    mdata = mu.MuData({"rna": rna, "atac": atac})
    return mdata

def main():
    parser = argparse.ArgumentParser(description="Build h5mu from ATAC + RNA")
    parser.add_argument("--samples", "-s", required=True, help="Path to samples.txt")
    parser.add_argument("--output", "-o", required=True, help="Full output filename for MuData (.h5mu)")
    args = parser.parse_args()

    with open(args.samples) as f:
        samples = [line.strip() for line in f if line.strip()]

    if len(samples) != 1:
        raise ValueError("This script only supports building one sample at a time.")

    sample = samples[0]
    mdata = build_mdata(sample)

    print(f"Saving MuData to {args.output} ...")
    mdata.write(args.output)
    print("✅ Done.")

if __name__ == "__main__":
    main()

