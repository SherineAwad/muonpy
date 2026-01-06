import pandas as pd
import subprocess
from scipy import sparse
from scipy.io import mmwrite
from intervaltree import IntervalTree
from collections import defaultdict
import os
import argparse

# --------------------------
# 1️⃣ Parse arguments
# --------------------------
parser = argparse.ArgumentParser(description="Preprocess ATAC fragments for Muon")
parser.add_argument("--samples", required=True, help="Path to samples.txt")
args = parser.parse_args()

# --------------------------
# 2️⃣ Read samples
# --------------------------
with open(args.samples) as f:
    samples = [line.strip() for line in f if line.strip()]

# --------------------------
# 3️⃣ Process each sample
# --------------------------
for sample in samples:
    print(f"\nProcessing sample: {sample}")

    fragments_file = f"{sample}_atac_fragments.tsv.gz"
    macs2_prefix = sample
    sample_out = sample

    # REMOVED: os.makedirs(sample, exist_ok=True)
    # REMOVED: os.chdir(sample)

    # --------------------------
    # 3a️⃣ Call peaks with MACS2
    # --------------------------
    print(f"Calling peaks with MACS2 for {sample}...")
    subprocess.run([
        "macs2", "callpeak",
        "-t", fragments_file,  # File is in current directory
        "-f", "BED",
        "-n", macs2_prefix,
        "--nomodel", "--shift", "-100", "--extsize", "200",
        "--keep-dup", "all", "--call-summits"
    ], check=True)

    peaks_file = f"{macs2_prefix}_peaks.narrowPeak"
    peaks_df = pd.read_csv(peaks_file, sep="\t", header=None)
    peaks_df['peak_name'] = peaks_df[0] + ":" + peaks_df[1].astype(str) + "-" + peaks_df[2].astype(str)
    peak_ids = peaks_df['peak_name'].values

    # Save peaks.tsv
    peaks_df['peak_name'].to_csv(f"{sample}_peaks.tsv", index=False, header=False)

    # --------------------------
    # 3b️⃣ Load fragments safely (first 4 columns)
    # --------------------------
    frags = pd.read_csv(fragments_file, sep="\t", header=None, usecols=[0,1,2,3],
                        names=["chr","start","end","barcode"])
    barcodes = frags['barcode'].unique()
    barcode_idx = {bc:i for i, bc in enumerate(barcodes)}

    # --------------------------
    # 3c️⃣ Build interval trees per chromosome
    # --------------------------
    trees = defaultdict(IntervalTree)
    for idx, row in peaks_df.iterrows():
        trees[row[0]][row[1]:row[2]] = idx

    # --------------------------
    # 3d️⃣ Build sparse peak ×cell matrix
    # --------------------------
    row_idx = []
    col_idx = []
    data = []

    for chr_, start, end, bc in zip(frags['chr'], frags['start'], frags['end'], frags['barcode']):
        for peak in trees[chr_][start:end]:
            row_idx.append(peak.data)
            col_idx.append(barcode_idx[bc])
            data.append(1)

    peak_cell_matrix = sparse.csr_matrix((data, (row_idx, col_idx)),
                                         shape=(len(peak_ids), len(barcodes)))

    # --------------------------
    # 3e️⃣ Save outputs
    # --------------------------
    mmwrite(f"{sample}_atac_counts.mtx", peak_cell_matrix)
    pd.Series(barcodes).to_csv(f"{sample}_barcodes.tsv", index=False, header=False)

    print(f"✅ Finished {sample}")
