#!/usr/bin/env python3
import argparse
import muon as mu
import scanpy as sc
import matplotlib.pyplot as plt
from muon import atac as ac
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="RNA+ATAC WNN UMAP (muon 0.1.7)")
    parser.add_argument("-i", "--input", required=True, help="Input filtered H5MU")
    parser.add_argument("-o", "--output", required=True, help="Output analysed H5MU")
    args = parser.parse_args()

    print(f"Reading {args.input}")
    mdata = mu.read_h5mu(args.input)

    # ---------------- RNA ----------------
    print("Processing RNA")
    rna = mdata.mod["rna"]

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=2000, subset=True)
    sc.pp.scale(rna)
    sc.tl.pca(rna, n_comps=50)

    # ---------------- ATAC ----------------
    print("Processing ATAC")
    atac = mdata.mod["atac"]
    
    # Filter out features with zero counts before TF-IDF
    nonzero_features = np.array(atac.X.sum(axis=0)).flatten() > 0
    if np.sum(~nonzero_features) > 0:
        print(f"Removing {np.sum(~nonzero_features)} features with zero counts")
        atac = atac[:, nonzero_features].copy()
        mdata.mod["atac"] = atac
    
    ac.pp.tfidf(atac)
    ac.tl.lsi(atac, n_comps=50)

    # ---------------- Joint UMAP ----------------
    print("Computing joint UMAP")
    
    # Create a joint representation by concatenating PCA and LSI
    # Take the first 20 components from each
    rna_pca = rna.obsm['X_pca'][:, :20]
    atac_lsi = atac.obsm['X_lsi'][:, :20]
    
    # Concatenate the representations
    joint_rep = np.concatenate([rna_pca, atac_lsi], axis=1)
    mdata.obsm['X_joint'] = joint_rep
    
    # Compute neighbors and UMAP on the joint representation
    sc.pp.neighbors(mdata, use_rep='X_joint')
    sc.tl.umap(mdata)

    # ---------------- Plot ----------------
    sc.pl.umap(mdata, show=False)
    out_png = args.input.replace(".h5mu", "_wnn_umap.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saving analysed MuData to {args.output}")
    mdata.write(args.output)

    print("Done.")

if __name__ == "__main__":
    main()
