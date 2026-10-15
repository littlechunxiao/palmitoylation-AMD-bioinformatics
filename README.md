# Reproducible analysis code — Palmitoylation and immune signatures in AMD retina

This folder supplies the code and derived data used to generate the results reported in the manuscript
"Palmitoylation-pathway expression and complement/immune signatures in age-related macular degeneration:
a donor-aware retinal transcriptomic analysis" (GSE29801, Agilent-014850 whole-genome 4×44K, GPL4133).

## Data source
- GEO accession: **GSE29801** (old GSE29801 series of 118 human neural-retina arrays from 62 donors).
- A two-color, common-reference, dye-swap design; series-matrix signal values are already log2-transformed.
- The final processed input read by the scripts is `data/retina_sample_metadata_clean.csv` (sample metadata,
  tissue region, donor grouping, diagnosis) together with the normalized log2 expression matrix
  (kept as `gene_expression_matrix.csv` in the raw data directory; reproduce it from GEO series matrix if needed).

## Environment
- Python ≥ 3.9. Run:
  ```
  pip install -r requirements.txt
  ```
- Scripts also call **Enrichr** (via `gseapy.enrichr`) and **STRING-DB v12** (via `stringdb`/HTTP) for enrichment and
  protein–protein interaction; these steps require network access. Immune signature genes are curated locally and do
  not call CIBERSORT itself.

## Execution order
Main results (revised, donor-aware pipeline):

1. `rev_deg_donor_cluster.py` — donor-clustered robust OLS (and mixed-effects confirmation) differential expression;
   outputs `tables/rev_palmitoylation_41genes_deg.csv` (Table 2) and whole-transcriptome DEG tables.
2. `rev_enrich_ppi.py` — KEGG/Reactome/GO enrichment of nominal DEGs and STRING-DB PPI; outputs
   `tables/rev_enrich_*.csv` (Table 5 / Supplementary).
3. `rev_immune_signature_scores.py` — 12 curated immune/complement signatures; outputs
   `tables/rev_immune_signature_scores.csv` (Table 4).
4. `rev_nested_lasso_donor_groupedCV.py` — donor-grouped **nested** cross-validation (outer *and* inner loops grouped
   by donor via GroupKFold, donor-level bootstrap 95% CI); outputs `tables/rev_nested_lasso_donor_summary.csv`
   and `_perfold.csv` (Table 3), and the pooled ROC figure.
5. `rev_build_figures.py` / `rev_build_suppfigures.py` — regenerate main (Figure 1–5) and supplementary (S1–S6) figures
   from the tables above.

## Important methodological note
Both the outer and the inner cross-validation loops of the LASSO classifier are grouped by donor (GroupKFold),
so samples from one donor never appear in both training and validation folds. The 95% CI of the pooled AUC is
estimated by **resampling donors** (bootstrap, n=2000). This is the corrected scheme reported in the manuscript.

## Inputs you must provide from GEO
The large normalized expression matrix is not shipped here for size. To reproduce:
download the GSE29801 series-matrix file from GEO, parse the log2 signal matrix, align features to GPL4133 gene
symbols, and save rows = genes / columns = samples as `gene_expression_matrix.csv` (index = gene symbol),
with sample columns matching `data/retina_sample_metadata_clean.csv`. `GPL4133_annotation_simple.csv` is provided
for the probe-to-symbol mapping.

## Derived data
Final tables shipped in `tables/` mirror those embedded in the manuscript and the submission 03_Tables folder.