import numpy as np
import pandas as pd
import scanpy as sc

df = pd.read_csv("./transcripts.csv")

scrna_annotations = pd.read_csv("./rawData_mouseStSt/annot_mouseStStAll.csv")
feature_table = pd.read_csv(
    "./rawData_mouseStSt/countTable_mouseStSt/features.tsv", header=None, index_col=0
)
barcodes = pd.read_csv(
    "./rawData_mouseStSt/countTable_mouseStSt/barcodes.tsv", header=None, index_col=0
)
count_matrix = sc.read_mtx("./rawData_mouseStSt/countTable_mouseStSt/matrix.mtx")

adata_scrna = sc.AnnData(X=count_matrix.X.T, var=feature_table, obs=barcodes)


del count_matrix

adata_scrna_filtered = adata_scrna[scrna_annotations.cell.values]
adata_scrna_filtered.obs = scrna_annotations

adata_scrna_filtered = adata_scrna_filtered[
    :, adata_scrna_filtered.var.index.isin(df.gene)
]

celltypes = sorted(adata_scrna_filtered.obs.annot.unique())
signatures = pd.DataFrame(
    index=adata_scrna_filtered.var.index, columns=celltypes
).fillna(0)

for i, celltype in enumerate(celltypes):
    signatures.loc[adata_scrna_filtered.var.index, celltype] = (
        adata_scrna_filtered.X[adata_scrna_filtered.obs.annot == celltype].mean(0).A1
    )

# signatures = signatures.loc[vis.genes]
signatures = np.log(1 + signatures / ((signatures**2).sum(0) ** 0.5))

signatures.to_csv("./signatures_liver_atlas.csv")
