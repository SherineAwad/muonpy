# **🔬 Scanpy + Muon for Multi-Omics**

Scanpy 🧬 is a Python toolkit for single-cell analysis, and **Muon** ✨ extends it to handle multiple modalities (RNA, protein, ATAC, etc.) in a single framework.  
It’s conceptually similar to **Seurat + Signac** in R 🖥️, allowing integrated analysis, visualization 📊, and interpretation of complex multi-omics datasets in a streamlined workflow 🚀.



# Summary of samples

| Sample | Overlapping barcodes (cells) | RNA features (genes) | ATAC features (peaks) |
|--------|------------------------------|----------------------|-----------------------|
| TH1    | 10,586                       | 32,285               | 218,774               |
| TH2    | 11,519                       | 32,285               | 229,725               |



# Before and after filtering 

### Stats 

| Stage | Cells | Peaks |
|------|-------|-------|
| Original | 22,105 | 447,355 |
| After cell filtering | 21,213 | 447,355 |
| After peak filtering | 21,213 | 447,244 |
| Final | 21,213 | 447,244 |


![Before](neurog2_qc_before_filtering.png?v=1)

![After](neurog2_qc_after_filtering.png?v=1)

