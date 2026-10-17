#!/usr/bin/env python3
"""
GSE29801 Data Preprocessing for Palmitoylation + AMD Analysis
All data is real - NO SIMULATION
"""
import gzip
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_PATH = "/app/data/所有对话/主对话/palmitoylation_amd_sci/v2_analysis/data/GSE29801_series_matrix.txt.gz"
OUT_DIR = "/app/data/所有对话/主对话/palmitoylation_amd_sci/real_analysis"
RESULTS_DIR = f"{OUT_DIR}/results"
TABLES_DIR = f"{OUT_DIR}/tables"
FIGURES_DIR = f"{OUT_DIR}/figures"
DATA_DIR = f"{OUT_DIR}/data"

# Set style for publication
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

def parse_series_matrix(filepath):
    """Parse GEO series matrix file and extract metadata + expression data."""
    print("Parsing GSE29801 series matrix...")
    
    metadata = {}
    sample_titles = []
    sample_gsm = []
    ch1_lines = []  # list of lists, each sublist is values for one characteristic
    expression_data = []
    probe_ids = []
    
    in_data_section = False
    header_parsed = False
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('!series_matrix_table_begin'):
                in_data_section = True
                continue
            
            if line.startswith('!series_matrix_table_end'):
                break
            
            if in_data_section:
                parts = line.split('\t')
                # Remove quotes
                parts = [p.strip('"') for p in parts]
                
                if not header_parsed:
                    # This is the header row with GSM IDs
                    sample_gsm = parts[1:]
                    header_parsed = True
                    continue
                
                probe_id = parts[0]
                values = parts[1:]
                # Convert to float
                try:
                    values = [float(v) for v in values]
                    probe_ids.append(probe_id)
                    expression_data.append(values)
                except ValueError:
                    continue
            
            elif line.startswith('!Sample_title'):
                parts = line.split('\t')
                sample_titles = [p.strip('"') for p in parts[1:]]
            
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = line.split('\t')
                char_values = [p.strip('"') for p in parts[1:]]
                ch1_lines.append(char_values)
            
            elif line.startswith('!Series_platform_id'):
                parts = line.split('\t')
                metadata['platform'] = parts[1].strip('"')
    
    # Create expression dataframe
    expr_df = pd.DataFrame(expression_data, index=probe_ids, columns=sample_gsm)
    expr_df.index.name = 'probe_id'
    
    # Create sample metadata
    sample_meta = pd.DataFrame({
        'gsm': sample_gsm,
        'sample_title': sample_titles
    }, index=sample_gsm)
    
    # Parse characteristics from ch1 lines
    # Each line has format "key: value" for each sample
    char_names = []
    char_data = {}
    
    for i, line_values in enumerate(ch1_lines):
        # Extract key from first non-empty value
        key = None
        for val in line_values:
            if ': ' in val:
                key = val.split(': ', 1)[0].strip()
                break
        if key:
            char_names.append(key)
            # Extract values
            extracted = []
            for val in line_values:
                if ': ' in val:
                    extracted.append(val.split(': ', 1)[1].strip())
                else:
                    extracted.append('')
            char_data[key] = extracted
    
    for key, values in char_data.items():
        if len(values) == len(sample_meta):
            sample_meta[key] = values
    
    metadata['n_samples'] = len(sample_gsm)
    metadata['n_probes'] = len(probe_ids)
    metadata['characteristics'] = char_names
    
    print(f"  Platform: {metadata.get('platform', 'Unknown')}")
    print(f"  Total samples: {metadata['n_samples']}")
    print(f"  Total probes: {metadata['n_probes']}")
    print(f"  Characteristics: {char_names}")
    
    return expr_df, sample_meta, metadata

def extract_retina_samples(expr_df, sample_meta):
    """Extract retina samples and their metadata."""
    print("\nExtracting retina samples...")
    
    # Identify retina samples by sample title
    retina_mask = sample_meta['sample_title'].str.startswith('Retina_')
    retina_samples = sample_meta[retina_mask].copy()
    
    print(f"  Total retina samples: {len(retina_samples)}")
    
    # Extract retina expression data
    retina_expr = expr_df[retina_samples['gsm'].values].copy()
    
    # Add tissue info
    retina_samples['tissue_type'] = 'Retina'
    
    # Parse region from tissue characteristic
    if 'tissue' in retina_samples.columns:
        retina_samples['region'] = retina_samples['tissue'].apply(
            lambda x: 'macular' if 'macular' in x.lower() else 
                      ('extramacular' if 'extramacular' in x.lower() else 'unknown')
        )
    
    # Get disease status
    if 'ocular disease' in retina_samples.columns:
        retina_samples['disease'] = retina_samples['ocular disease']
    else:
        retina_samples['disease'] = 'unknown'
    
    # Get AMD classification
    if 'amd classification' in retina_samples.columns:
        retina_samples['amd_class'] = retina_samples['amd classification']
    else:
        retina_samples['amd_class'] = 'unknown'
    
    # Print summary
    print(f"\n  Disease distribution:")
    print(retina_samples['disease'].value_counts())
    
    print(f"\n  AMD classification distribution:")
    print(retina_samples['amd_class'].value_counts())
    
    if 'region' in retina_samples.columns:
        print(f"\n  Region distribution:")
        print(retina_samples['region'].value_counts())
    
    return retina_expr, retina_samples

def define_groups(sample_meta):
    """Define normal vs AMD groups for comparison."""
    print("\nDefining comparison groups...")
    
    # Normal: amd_class == 'normal'
    normal_mask = sample_meta['amd_class'] == 'normal'
    
    # AMD: any AMD classification (excluding pre-AMD for main comparison)
    amd_classes = ['dry AMD', 'GA', 'CNV', 'GA/CNV', 'clinical AMD diagnosis']
    amd_mask = sample_meta['amd_class'].isin(amd_classes)
    
    # Pre-AMD
    pre_amd_classes = ['MD1', 'MD2']
    pre_amd_mask = sample_meta['amd_class'].isin(pre_amd_classes)
    
    sample_meta['group'] = 'other'
    sample_meta.loc[normal_mask, 'group'] = 'Normal'
    sample_meta.loc[amd_mask, 'group'] = 'AMD'
    sample_meta.loc[pre_amd_mask, 'group'] = 'Pre-AMD'
    
    print(f"  Normal: {normal_mask.sum()}")
    print(f"  AMD: {amd_mask.sum()}")
    print(f"  Pre-AMD: {pre_amd_mask.sum()}")
    print(f"  Other/Unknown: {(sample_meta['group'] == 'other').sum()}")
    
    return sample_meta

def log2_transform(expr_df):
    """Log2 transform expression data if not already done."""
    # Check if data is already log-transformed
    # If max value > 100, probably raw intensities
    if expr_df.max().max() > 100:
        print("\nApplying log2 transformation...")
        # Add small offset to avoid log(0)
        expr_log = np.log2(expr_df + 1)
        return expr_log
    else:
        print("\nData appears already log-transformed, skipping log2.")
        return expr_df.copy()

def quantile_normalization(expr_df):
    """Perform quantile normalization."""
    print("\nPerforming quantile normalization...")
    
    # Sort each column
    sorted_df = expr_df.apply(lambda x: x.sort_values().values)
    # Compute row means (rank means)
    rank_means = sorted_df.mean(axis=1)
    # Replace each column's sorted values with rank means
    norm_df = expr_df.copy()
    for col in norm_df.columns:
        # Get the order of original values
        rank_order = expr_df[col].rank(method='first').astype(int) - 1
        norm_df[col] = rank_means.values[rank_order]
    
    return norm_df

def differential_expression(expr_df, sample_meta, group_col='group', control='Normal', case='AMD'):
    """
    Perform differential expression analysis using t-test with multiple testing correction.
    """
    print(f"\nPerforming differential expression analysis: {case} vs {control}...")
    
    # Get sample groups
    control_samples = sample_meta[sample_meta[group_col] == control].index
    case_samples = sample_meta[sample_meta[group_col] == case].index
    
    print(f"  Control samples: {len(control_samples)}")
    print(f"  Case samples: {len(case_samples)}")
    
    if len(control_samples) < 2 or len(case_samples) < 2:
        print("  WARNING: Not enough samples for DE analysis!")
        return pd.DataFrame()
    
    control_expr = expr_df[control_samples]
    case_expr = expr_df[case_samples]
    
    # Calculate statistics
    control_mean = control_expr.mean(axis=1)
    case_mean = case_expr.mean(axis=1)
    
    # Log2 fold change
    log2fc = case_mean - control_mean
    
    # T-test (Welch's t-test for unequal variance)
    t_stats, p_values = stats.ttest_ind(case_expr.T, control_expr.T, equal_var=False)
    
    # Handle NaN p-values (constant expression)
    p_values = np.nan_to_num(p_values, nan=1.0)
    
    # Multiple testing correction (Benjamini-Hochberg)
    reject, adj_pvals, _, _ = multipletests(p_values, method='fdr_bh')
    
    # Create results dataframe
    deg_results = pd.DataFrame({
        'log2FC': log2fc,
        'p_value': p_values,
        'adj_p_value': adj_pvals,
        'control_mean': control_mean,
        'case_mean': case_mean,
        'significant_adj_p05': adj_pvals < 0.05,
        'significant_adj_p05_fc05': (adj_pvals < 0.05) & (np.abs(log2fc) > 0.5)
    })
    
    n_sig = (adj_pvals < 0.05).sum()
    n_up = ((adj_pvals < 0.05) & (log2fc > 0)).sum()
    n_down = ((adj_pvals < 0.05) & (log2fc < 0)).sum()
    
    n_sig_fc = ((adj_pvals < 0.05) & (np.abs(log2fc) > 0.5)).sum()
    
    print(f"  Significant DEGs (adj p < 0.05): {n_sig}")
    print(f"    Up-regulated: {n_up}")
    print(f"    Down-regulated: {n_down}")
    print(f"  Significant (adj p < 0.05 AND |log2FC| > 0.5): {n_sig_fc}")
    
    return deg_results

def plot_volcano(deg_results, output_path, title="Volcano Plot"):
    """Create volcano plot."""
    if deg_results.empty:
        print("  No data for volcano plot.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # -log10 p-value
    neg_log_p = -np.log10(deg_results['adj_p_value'].clip(lower=1e-20))
    
    # Colors
    colors = []
    for _, row in deg_results.iterrows():
        if row['adj_p_value'] < 0.05 and row['log2FC'] > 0.5:
            colors.append('#D55E00')  # Up - vermillion
        elif row['adj_p_value'] < 0.05 and row['log2FC'] < -0.5:
            colors.append('#0072B2')  # Down - blue
        else:
            colors.append('#999999')  # Not significant - gray
    
    ax.scatter(deg_results['log2FC'], neg_log_p, c=colors, s=10, alpha=0.6)
    
    # Add threshold lines
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=0.8, label='adj p = 0.05')
    ax.axvline(x=0.5, color='black', linestyle=':', linewidth=0.8, label='log2FC = ±0.5')
    ax.axvline(x=-0.5, color='black', linestyle=':', linewidth=0.8)
    
    ax.set_xlabel('log2 Fold Change (AMD / Normal)')
    ax.set_ylabel('-log10 (Adjusted p-value)')
    ax.set_title(title)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D55E00', markersize=8, label='Up-regulated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0072B2', markersize=8, label='Down-regulated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=8, label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Volcano plot saved: {output_path}")

def plot_pca(expr_df, sample_meta, output_path, title="PCA Plot"):
    """Create PCA plot."""
    from sklearn.decomposition import PCA
    
    # Transpose for PCA (samples x genes)
    expr_t = expr_df.T
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(expr_t)
    
    # Create dataframe
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'group': sample_meta.loc[expr_t.index, 'group'].values,
        'sample': expr_t.index
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'Normal': '#009E73', 'AMD': '#D55E00', 'Pre-AMD': '#E69F00', 'other': '#999999'}
    
    for group in sorted(pca_df['group'].unique()):
        mask = pca_df['group'] == group
        ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'], 
                   c=colors.get(group, '#999999'), label=group, s=40, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  PCA plot saved: {output_path}")

def main():
    # Step 1: Parse data
    expr_df, sample_meta, metadata = parse_series_matrix(DATA_PATH)
    
    # Print sample meta columns
    print("\nSample metadata columns:")
    print(sample_meta.columns.tolist())
    
    # Step 2: Extract retina samples
    retina_expr, retina_meta = extract_retina_samples(expr_df, sample_meta)
    
    # Step 3: Define groups
    retina_meta = define_groups(retina_meta)
    
    # Step 4: Log2 transform
    retina_expr_log = log2_transform(retina_expr)
    
    # Step 5: Quantile normalization
    retina_expr_norm = quantile_normalization(retina_expr_log)
    
    # Save processed data
    retina_expr_norm.to_csv(f"{DATA_DIR}/retina_expression_normalized.csv")
    retina_meta.to_csv(f"{DATA_DIR}/retina_sample_metadata.csv")
    
    print(f"\nProcessed data saved to {DATA_DIR}")
    print(f"Expression matrix: {retina_expr_norm.shape[0]} probes x {retina_expr_norm.shape[1]} samples")
    
    # Step 6: PCA
    print("\nGenerating PCA plot...")
    plot_pca(retina_expr_norm, retina_meta, 
             f"{FIGURES_DIR}/pca_retina_all.png",
             "PCA of GSE29801 Retina Samples")
    
    # Step 7: Differential expression (AMD vs Normal)
    # Use only Normal and AMD samples for DE analysis
    de_samples = retina_meta[retina_meta['group'].isin(['Normal', 'AMD'])]
    de_expr = retina_expr_norm[de_samples.index]
    
    if len(de_samples) > 0:
        deg_results = differential_expression(de_expr, de_samples)
        
        # Save DEG results
        if not deg_results.empty:
            deg_results.to_csv(f"{TABLES_DIR}/all_degs_amd_vs_normal.csv")
            
            # Step 8: Volcano plot
            print("\nGenerating volcano plot...")
            plot_volcano(deg_results, 
                         f"{FIGURES_DIR}/volcano_amd_vs_normal.png",
                         "Volcano Plot: AMD vs Normal Retina (GSE29801)")
    else:
        print("\nNo Normal/AMD samples for DE analysis!")
    
    # Step 9: Subtype analysis
    subtypes = ['dry AMD', 'GA', 'CNV']
    for subtype in subtypes:
        subtype_mask = (retina_meta['amd_class'] == subtype) | (retina_meta['amd_class'] == 'normal')
        subtype_samples = retina_meta[subtype_mask]
        
        n_subtype = (retina_meta['amd_class'] == subtype).sum()
        n_normal = (retina_meta['amd_class'] == 'normal').sum()
        
        if n_subtype >= 3 and n_normal >= 3:
            print(f"\n--- {subtype} vs Normal ---")
            subtype_meta = subtype_samples.copy()
            subtype_meta['group'] = subtype_meta['amd_class'].apply(
                lambda x: 'Normal' if x == 'normal' else 'AMD'
            )
            subtype_expr = retina_expr_norm[subtype_meta.index]
            subtype_deg = differential_expression(subtype_expr, subtype_meta)
            
            if not subtype_deg.empty:
                fname = f"degs_{subtype.lower().replace(' ', '_')}_vs_normal.csv"
                subtype_deg.to_csv(f"{TABLES_DIR}/{fname}")
                
                # Volcano plot
                volc_fname = f"volcano_{subtype.lower().replace(' ', '_')}_vs_normal.png"
                plot_volcano(subtype_deg,
                           f"{FIGURES_DIR}/{volc_fname}",
                           f"Volcano Plot: {subtype} vs Normal Retina")
    
    print("\n=== Data Preprocessing Complete ===")

if __name__ == "__main__":
    main()
