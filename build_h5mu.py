#!/usr/bin/env python3
import argparse
import pandas as pd
import muon as mu
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread
import scipy.sparse

def build_mdata(sample):
    print(f"\nProcessing sample: {sample}")

    peaks_file = f"{sample}_peaks.tsv"
    barcodes_file = f"{sample}_barcodes.tsv"
    counts_file = f"{sample}_atac_counts.mtx"

    peak_ids = pd.read_csv(peaks_file, header=None)[0].values
    barcodes = pd.read_csv(barcodes_file, header=None)[0].values
    peak_cell_matrix = mmread(counts_file).tocsr()

    atac = AnnData(X=peak_cell_matrix.T)
    atac.obs_names = barcodes
    atac.var_names = peak_ids
    atac.var_names_make_unique()
    atac.obs['sample'] = sample

    rna_h5 = f"{sample}_filtered_feature_bc_matrix.h5"
    rna = sc.read_10x_h5(rna_h5)
    rna.var_names_make_unique()
    rna.obs['sample'] = sample

    common_barcodes = set(rna.obs_names).intersection(set(atac.obs_names))
    
    if len(common_barcodes) == 0:
        print(f"⚠️  WARNING: No common barcodes found between RNA and ATAC!")
        print(f"   RNA barcodes: {len(rna.obs_names)}")
        print(f"   ATAC barcodes: {len(atac.obs_names)}")
    else:
        print(f"   Found {len(common_barcodes)} common barcodes")
    
    rna = rna[rna.obs_names.isin(common_barcodes)].copy()
    atac = atac[atac.obs_names.isin(common_barcodes)].copy()
    
    rna = rna[atac.obs_names].copy()

    mdata = mu.MuData({"rna": rna, "atac": atac})
    mdata.obs['sample'] = sample
    
    print(f"   RNA shape: {rna.shape}")
    print(f"   ATAC shape: {atac.shape}")
    
    return mdata

def main():
    parser = argparse.ArgumentParser(description="Build h5mu from ATAC + RNA")
    parser.add_argument("--samples", "-s", required=True, help="Path to samples.txt")
    parser.add_argument("--output", "-o", required=True, help="Full output filename for MuData (.h5mu)")
    args = parser.parse_args()

    with open(args.samples) as f:
        samples = [line.strip() for line in f if line.strip()]

    # Process all samples
    mdatas = []
    for sample in samples:
        mdata = build_mdata(sample)
        mdatas.append(mdata)

    if len(mdatas) == 1:
        print(f"Saving MuData to {args.output} ...")
        mdatas[0].write(args.output)
    else:
        print(f"Combining {len(mdatas)} samples into one h5mu...")
        
        # Combine RNA - genes should be same across samples
        rna_matrices = []
        rna_obs = []
        for m in mdatas:
            rna_matrices.append(m['rna'].X)
            rna_obs.append(m['rna'].obs)
        
        rna_combined = AnnData(
            X=scipy.sparse.vstack(rna_matrices),
            obs=pd.concat(rna_obs, ignore_index=True),
            var=mdatas[0]['rna'].var
        )
        
        # Combine ATAC - need union of peaks
        all_peaks = set()
        for m in mdatas:
            all_peaks.update(m['atac'].var_names)
        all_peaks = sorted(list(all_peaks))
        
        atac_matrices = []
        atac_obs = []
        for m in mdatas:
            # Map peaks to combined peak set
            peak_to_idx = {peak: i for i, peak in enumerate(all_peaks)}
            m_peaks = m['atac'].var_names
            
            # Create mapping matrix
            row = []
            col = []
            data = []
            for i, peak in enumerate(m_peaks):
                col_idx = peak_to_idx[peak]
                # Get all non-zero values for this peak
                col_data = m['atac'].X[:, i]
                nonzero = col_data.nonzero()
                row.extend(nonzero[0])
                col.extend([col_idx] * len(nonzero[0]))
                data.extend(col_data.data)
            
            # Create sparse matrix
            matrix = scipy.sparse.csr_matrix(
                (data, (row, col)), 
                shape=(m['atac'].shape[0], len(all_peaks))
            )
            atac_matrices.append(matrix)
            atac_obs.append(m['atac'].obs)
        
        atac_combined = AnnData(
            X=scipy.sparse.vstack(atac_matrices),
            obs=pd.concat(atac_obs, ignore_index=True),
            var=pd.DataFrame(index=all_peaks)
        )
        
        # Create combined MuData
        combined_mdata = mu.MuData({"rna": rna_combined, "atac": atac_combined})
        
        print(f"Combined shape: {combined_mdata.shape}")
        print(f"Saving combined MuData to {args.output} ...")
        combined_mdata.write(args.output)
    
    print("✅ Done.")

if __name__ == "__main__":
    main()
