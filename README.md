# **🔬 Scanpy + Muon for Multi-Omics**

Scanpy 🧬 is a Python toolkit for single-cell analysis, and **Muon** ✨ extends it to handle multiple modalities (RNA, protein, ATAC, etc.) in a single framework.  
It’s conceptually similar to **Seurat + Signac** in R 🖥️, allowing integrated analysis, visualization 📊, and interpretation of complex multi-omics datasets in a streamlined workflow 🚀.



# Summary of samples

| Sample | Overlapping barcodes (cells) | RNA features (genes) | ATAC features (peaks) |
|--------|------------------------------|----------------------|-----------------------|
| TH1    | 10,586                       | 32,285               | 218,774               |
| TH2    | 11,519                       | 32,285               | 229,725               |



# Before and after filtering 

![rna](neurog2_rna_qc_before_filtering.png?v=2)
![atac](neurog2_atac_qc_before_filtering.png?v=2)

# After Filtering 

![rna](neurog2_rna_qc_after_filtering.png?v=2)
![atac](neurog2_atac_qc_after_filtering.png?v=2)

### Stats

| Modality | Metric | Before Filtering | After Filtering | Notes |
|--------|--------|------------------|-----------------|-------|
| **RNA** | Cells | 22,105 | 19,340 | Initial cell filtering |
| **RNA** | Cells (final aligned) | 22,105 | 18,380 | 83.1% retained |
| **RNA** | Genes | 32,285 | 21,679 | min_cells = 10 |
| **ATAC** | Cells | 22,105 | 21,097 | Initial cell filtering |
| **ATAC** | Cells (final aligned) | 22,105 | 18,380 | 83.1% retained |
| **ATAC** | Peaks | 447,355 | 447,228 | min_cells = 10 |


#### UMAP 


![rna](neurog2_filtered_umap_rna_only.png?v=1)
![atac](neurog2_filtered_umap_atac_only.png?v=1)
![overlapped](neurog2_filtered_umap_rna_atac_overlapped.png?v=1)



