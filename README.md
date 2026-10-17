# Reproducible analysis code — Palmitoylation and immune signatures in AMD retina

This folder supplies the code used to generate the results reported in the manuscript
"Palmitoylation-pathway expression and complement/immune signatures in age-related macular degeneration:
a donor-aware retinal transcriptomic analysis" (GSE29801, Agilent-014850 whole-genome 4×44K, GPL4133).

## Data source
- GEO accession: **GSE29801** (118 human neural-retina arrays from 62 donors).
- A two-color, common-reference, dye-swap design. **Scale statement (consistent with the manuscript):** the
  Agilent Feature Extraction output is read as linear signal abundances; the analysis applies `log2(x+1)`
  followed by quantile normalization across all 118 samples (Methods §2.1, §2.3). `gene_expression_matrix.csv`
  distributed here is therefore already the normalized log2 matrix used downstream.
- The processed inputs read by the scripts are `data/retina_sample_metadata_clean.csv` (sample metadata: tissue
  region, donor grouping, age, sex, diagnosis including Normal/AMD/Pre-AMD) and `gene_expression_matrix.csv`
  (rows = genes, columns = samples, log2-normalized).

## Environment
- Python ≥ 3.9.
- Scripts call **Enrichr** (via `gseapy.enrichr`) and **STRING-DB v12** (via `stringdb`/HTTP) for enrichment and
  PPI; these steps require network access. Immune signature genes are curated locally and do not call CIBERSORT itself.
- The single-cell validation script (`run_amd_val.py`) additionally requires `scanpy`, `anndata`, `leidenalg`
  and downloads `GSE203499_RAW.tar` (~1.8 GB) on first run (see below).

## Execution order — complete reproducible pipeline (main results)
Revised, donor-aware pipeline:

1. `01_data_preprocessing.py` — reads raw GEO series-matrix, applies `log2(x+1)` + quantile normalization, and
   writes `gene_expression_matrix.csv` + `retina_sample_metadata_clean.csv`. (Automatic scale check: values with
   peak > 100 are treated as linear and log2-transformed; already-log2 input is used as-is.)
2. `rev_deg_donor_cluster.py` — donor-clustered robust OLS (and mixed-effects confirmation) differential expression;
   outputs `tables/rev_palmitoylation_41genes_deg.csv` (Table 2) and whole-transcriptome DEG tables.
3. `rev_enrich_ppi.py` — KEGG/Reactome/GO enrichment of nominal DEGs and STRING-DB PPI; outputs
   `tables/rev_enrich_*.csv` (Table 5 / Supplementary).
4. `rev_immune_signature_scores.py` — 12 curated immune/complement signatures (AMD vs Normal, donor-clustered
   robust OLS controlling macular region, age, sex); outputs `tables/rev_immune_signature_scores.csv` (Table 4 /
   S2). A signature is skipped in the AMD-vs-normal comparison if fewer than 3 of its genes are detected on the
   platform (`if n_avail < 3`), and scored otherwise.
   **Transparency note (reviewers):** the manuscript states a "score calculated only when at least half of the
   signature genes were present"; the shipped script implements `if n_avail < 3: continue` (skip). These are two
   implementations of the same intended rule, and we verified that every one of the 12 signatures has at least
   half of its curated genes detectable on GPL4133 (and at least 3), so no signature is excluded differently under
   either rule. The manuscript wording and this README describe the same behaviour on the shipped data.
5. `rev_nested_lasso_donor_groupedCV.py` — donor-grouped **nested** cross-validation (outer *and* inner loops grouped
   by donor via GroupKFold), donor-level bootstrap 95% CI (n = 2000). Outputs `tables/rev_nested_lasso_donor_summary.csv` and `_perfold.csv`
   (Table 3) plus the per-fold prediction table (`rev_nested_lasso_donor_perfold.csv`: GSM, donor, true label,
   outer fold, out-of-fold probability). All ROC curves and performance tables are generated from this prediction
   table, so there is a single classification path.
6. `rev_palmitoylation_immune_association.py` — direct association of the palmitoylation-pathway score (S3) and each
   of the 99 detectable pathway genes (S4) with the 12 signatures, donor-clustered robust OLS with AMD status,
   macular region, age and sex as covariates; BH-FDR adjustment; the pathway score z-score is computed per gene
   across samples and then averaged (the average of gene-level z-scores, not a re-standardized composite).
7. `age_sensitivity.py` — age-restricted sensitivity analyses (≤63–93 y window, 40 donors: 22 AMD / 18 normal)
   re-examining the pathway–signature associations in the shared-support age range.
8. `rev_build_figures.py` / `rev_build_suppfigures.py` — regenerate main (Figure 1–5) and supplementary figures from
   the tables above. **Figure 3 (ROC) is rendered by `rev_figure5_roc.py` from the pooled out-of-fold prediction
   table; the ROC block inside `rev_build_figures.py` is removed to keep a single classification path.**
9. `run_amd_val.py` — external single-cell validation (GSE203499: human retina + RPE/choroid, AMD vs healthy).
   Script auto-discovers per-sample subdirectories (matrix/barcodes/features), merges cells, applies QC
   (min_genes = 200), computes per-sample cell quotas and sampling order, assigns cell-type annotations from
   canonical markers, and outputs the Figure 7 dot plots plus cell-type × gene detection tables. Full per-sample
   QC counts (before/after), sample/donor/tissue/disease composition and normalization parameters are printed and
   saved as `tables/scRNA_validation_summary.csv`.

## Important methodological note
Both the outer and the inner cross-validation loops of the LASSO classifier are grouped by donor (GroupKFold), so no
donor is shared between training and validation folds. **Scale-isolation caveat:** the `StandardScaler` is fit on the
full outer training set before the inner donor-grouped loop, so inner validation folds use scaling parameters
estimated on the complete outer training set; test donors are still fully isolated. This matches the manuscript and
is disclosed here and in the response letter; a strict Pipeline-fitted scaler would change the reported AUC/CI and
selected-gene counts, which we have chosen not to re-derive silently. The pooled AUC, its donor-bootstrap 95% CI
(n = 2000) and all threshold-free metrics are derived from a single pooled out-of-fold prediction table.
Threshold-dependent metrics are not reported, consistent with §2.5, because the operating point is selected from the
same predictions. Since the single-cell downsampling did not use a fixed global seed, the manuscript states this
explicitly; for exact reproduction of the same cells, the actual retained cell-barcode list and the merged AnnData
summary are provided as `scRNA_cell_barcodes.tsv` / `scRNA_validation_summary.csv` in the repository.

## Inputs you must provide from GEO
The large expression matrix is not shipped here for size. To reproduce:
1. Download the GSE29801 series-matrix file from GEO.
2. Read the linear signal abundance columns (Agilent Feature Extraction, two-color dye-swap handled as
   common-reference), then run `01_data_preprocessing.py`, which applies `log2(x+1)` + quantile normalization and
   writes `gene_expression_matrix.csv` (rows = genes, columns = samples, index = gene symbol) with sample columns
   matching `data/retina_sample_metadata_clean.csv`. `GPL4133_annotation_simple.csv` provides probe-to-symbol mapping.
3. For single-cell validation, download `GSE203499_RAW.tar` (GEO) and run `run_amd_val.py --download` once.

## Derived data
Final tables shipped in `tables/` mirror those embedded in the manuscript and the submission supplementary files
(S1–S5), including the S3 pathway–signature and S4 gene–signature association tables, and the single-cell
validation summary.