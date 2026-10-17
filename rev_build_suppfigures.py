#!/usr/bin/env python3
"""
Build SUPPLEMENTARY figures from revised real analysis outputs.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import os

BASE = "/Coze/Drive/科研大帝/所有对话/主对话/palmitoylation_amd_sci/real_analysis"
TAB = f"{BASE}/tables"
FIGD = f"{BASE}/figures_rebuilt"
FIG = FIGD
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','DejaVu Sans'],
    'axes.linewidth':0.8,'axes.edgecolor':'#333','xtick.direction':'out','ytick.direction':'out',
    'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'figure.dpi':300,
})
AMD_C='#C0392B'; NORM_C='#2E86C1'; PRE_C='#27AE60'

expr=pd.read_csv(f"{BASE}/data/gene_expression_matrix.csv", index_col=0)
meta=pd.read_csv(f"{BASE}/data/retina_sample_metadata_clean.csv")

# ===== S1: PCA (all 118, colored by group) =====
X=expr.T.values
meta_idx=[g for g in meta['gsm'] if g in expr.columns]
X=expr[meta_idx].T.values
grp=meta.set_index('gsm').loc[meta_idx,'group'].astype(str)
pca=PCA(n_components=2)
pc=pca.fit_transform(X)
pc1e,pc2e=pca.explained_variance_ratio_
cmap={'Normal':NORM_C,'AMD':AMD_C,'Pre-AMD':PRE_C}
fig,ax=plt.subplots(figsize=(6.5,5.5))
for g in ['Normal','AMD','Pre-AMD']:
    m=grp==g
    ax.scatter(pc[m,0],pc[m,1],s=16,color=cmap[g],alpha=0.75,edgecolors='none',label=f'{g} (n={m.sum()})')
ax.set_xlabel(f'PC1 ({pc1e*100:.1f}%)'); ax.set_ylabel(f'PC2 ({pc2e*100:.1f}%)')
ax.set_title('PCA of 118 retinal samples (GSE29801)')
ax.legend(frameon=False, loc='best')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG}/S1_PCA_revised.png",dpi=300); plt.savefig(f"{FIG}/S1_PCA_revised.pdf"); plt.close()
print("S1 done")

# ===== S2: boxplots of 4 nominal palm genes =====
PALM_NOM=['ZDHHC16','ZDHHC6','PPT2','LYPLAL1']
amd_samples=[g for g in meta_idx if grp[g]=='AMD']
nor_samples=[g for g in meta_idx if grp[g]=='Normal']
fig,axes=plt.subplots(1,4,figsize=(13,3.6))
for ax,g in zip(axes,PALM_NOM):
    v=expr.loc[g, nor_samples+amd_samples].astype(float)
    data=[v[g2] for g2 in nor_samples],[v[g2] for g2 in amd_samples]
    bp=ax.boxplot(data,patch_artist=True,widths=0.5)
    ax.set_xticks([1,2]); ax.set_xticklabels(['Normal','AMD'])
    for patch,col in zip(bp['boxes'],[NORM_C,AMD_C]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_title(g,fontsize=10,fontweight='bold')
    ax.set_ylabel('log2 expression')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Expression of nominally significant palmitoylation genes (donor-clustered)',fontsize=10)
plt.tight_layout(); plt.savefig(f"{FIG}/S2_boxplots_palm_revised.png",dpi=300); plt.savefig(f"{FIG}/S2_boxplots_palm_revised.pdf"); plt.close()
print("S2 done")

# ===== S3: KEGG up bubble =====
def bubble(f,col,xlab,tit,fname):
    d=pd.read_csv(f)
    d=d.sort_values('Adjusted P-value').head(12)
    d['-log10q']=-np.log10(d['Adjusted P-value'].replace(0,1e-300))
    y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(7,5.2))
    sc=ax.scatter(d[col].values, y, s=d['-log10q'].values*60, c=d['-log10q'].values, cmap='viridis', vmin=max(d['-log10q'].min(),0.1), vmax=d['-log10q'].max())
    ax.set_yticks(y); ax.set_yticklabels(d['Term'].str[:45], fontsize=7.5)
    ax.set_xlabel(xlab)
    ax.set_title(tit,fontsize=9)
    cbar=plt.colorbar(sc,ax=ax,fraction=0.03); cbar.set_label('-log$_{10}$(q)',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{FIG}/{fname}.png",dpi=300); plt.savefig(f"{FIG}/{fname}.pdf"); plt.close()
bubble(f"{TAB}/rev_enrich_upregulated_KEGG.csv",'Odds Ratio','Odds ratio','KEGG enrichment of upregulated DEGs (n=1422 nominal)', 'S3_KEGG_up_revised')
print("S3 done")

# ===== S4: Reactome up (complement highlight) =====
bubble(f"{TAB}/rev_enrich_upregulated_Reactome.csv",'Odds Ratio','Odds ratio','Reactome enrichment of upregulated DEGs', 'S4_Reactome_up_revised')
print("S4 done")

# ===== S5: KEGG down (oxphos) =====
bubble(f"{TAB}/rev_enrich_downregulated_KEGG.csv",'Odds Ratio','Odds ratio','KEGG enrichment of downregulated DEGs', 'S5_KEGG_down_revised')
print("S5 done")

# ===== S6: immune signature heatmap =====
isc=pd.read_csv(f"{TAB}/rev_immune_signature_scores.csv")
isc=isc.sort_values('signature')
fig,ax=plt.subplots(figsize=(6.2,4.6))
vals=isc[['mean_Normal','mean_AMD']].values
im=ax.imshow(vals.T,aspect='auto',cmap='RdBu_r',vmin=-0.3,vmax=0.3)
ax.set_xticks(range(len(isc))); ax.set_xticklabels(isc['signature'],fontsize=6.5,rotation=45,ha='right')
ax.set_yticks([0,1]); ax.set_yticklabels(['Normal','AMD'])
cbar=plt.colorbar(im,ax=ax,fraction=0.03); cbar.set_label('Mean signature score',fontsize=8)
ax.set_title('Signature enrichment scores (mean z-score)')
plt.tight_layout(); plt.savefig(f"{FIG}/S6_immune_heatmap_revised.png",dpi=300); plt.savefig(f"{FIG}/S6_immune_heatmap_revised.pdf"); plt.close()
print("S6 done")

print("ALL SUPPLEMENTARY FIGURES DONE")