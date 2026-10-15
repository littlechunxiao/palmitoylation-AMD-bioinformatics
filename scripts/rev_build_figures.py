#!/usr/bin/env python3
"""
Rebuild all main figures from REVISED real analysis outputs.
Main figure set (Frontiers-style), 300 dpi, no AI watermark, consistent styling.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

BASE = "/Coze/Drive/科研大帝/所有对话/主对话/palmitoylation_amd_sci/real_analysis"
TAB = f"{BASE}/tables"
FIG = f"{BASE}/figures_rebuilt"
os.makedirs(FIG, exist_ok=True)

# consistent style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})
AMD_C = '#C0392B'
NORM_C = '#2E86C1'
PALM = ['ZDHHC1','ZDHHC2','ZDHHC3','ZDHHC4','ZDHHC5','ZDHHC6','ZDHHC7','ZDHHC8','ZDHHC9','ZDHHC11','ZDHHC12','ZDHHC13','ZDHHC14','ZDHHC15','ZDHHC16','ZDHHC17','ZDHHC18','ZDHHC19','ZDHHC20','ZDHHC21','ZDHHC22','ZDHHC23','ZDHHC24','ZDHHC25','PPT1','PPT2','LYPLA1','LYPLA2','ABHD10','ABHD12','ABHD16A','ABHD17A','ABHD17B','ABHD17C','BST2','GCP16','LYPLAL1','RAB7A','RAB7B','SELENOK','SELENOS']

# ============ Figure 1: Volcano ============
deg = pd.read_csv(f"{TAB}/rev_DEG_amd_vs_normal_donorcluster.csv")
deg['direction'] = np.where(deg['log2fc'] > 0, 'up', 'down')
nom = pd.read_csv(f"{TAB}/rev_palmitoylation_41genes_deg.csv")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
neglogp = -np.log10(deg['p'].replace(0, np.nan))
ax.scatter(deg['log2fc'], neglogp, s=10, c='#BFC9CA', alpha=0.55, edgecolors='none', zorder=2, label='Non-significant')
# nominal DEGs (p<0.05) pale
nom_p = deg[deg['p'] < 0.05]
# up/down nominal without FDR
up = nom_p[nom_p['log2fc'] > 0]
dn = nom_p[nom_p['log2fc'] < 0]
ax.scatter(up['log2fc'], -np.log10(up['p']), s=16, c='#E67E22', alpha=0.7, edgecolors='none', zorder=3, label='Nominal \u2191 (p<0.05)')
ax.scatter(dn['log2fc'], -np.log10(dn['p']), s=16, c='#5D6D7E', alpha=0.7, edgecolors='none', zorder=3, label='Nominal \u2193 (p<0.05)')
# FDR significant: ZFP42
fdr = deg[deg['fdr'] < 0.05]
ax.scatter(fdr['log2fc'], -np.log10(fdr['p']), s=30, marker='D', c='#5DADE2', edgecolors='#1B4F72', linewidths=0.6, zorder=5, label='FDR < 0.05')
for _,r in fdr.iterrows():
    ax.annotate(r['gene'], (r['log2fc'], -np.log10(r['p'])), xytext=(4,-2), textcoords='offset points', fontsize=7, color='#1B4F72', fontweight='bold')
# nominal palm genes annotate
pm_sub = nom[(nom['p']<0.05)]
for _,r in pm_sub.iterrows():
    ax.scatter(r['log2fc'], -np.log10(max(r['p'],1e-10)), s=40, marker='o', facecolors='none', edgecolors='#C0392B', linewidths=0.9, zorder=6)
    ax.annotate(r['gene'], (r['log2fc'], -np.log10(max(r['p'],1e-10))), xytext=(4,-2), textcoords='offset points', fontsize=7, color='#C0392B', fontweight='bold')
ax.axhline(-np.log10(0.05), color='#999999', ls='--', lw=0.7)
ax.axvline(0, color='#999999', ls='-', lw=0.6)
ax.set_xlabel('log$_2$ fold change (AMD \u2212 Normal)')
ax.set_ylabel('-log$_{10}$(p-value)')
ax.set_xlim(-1.5, 1.5)
ax.legend(frameon=False, loc='upper right', fontsize=6.5)
ax.set_title('Volcano plot of differentially expressed genes\n(donor-clustered robust regression, retina)', fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Figure1_volcano_revised.png", dpi=300)
plt.savefig(f"{FIG}/Figure1_volcano_revised.pdf")
plt.close()
print("Figure1 done")

# ============ Figure 2: Heatmap palm 33 genes ============
expr = pd.read_csv(f"{BASE}/data/gene_expression_matrix.csv", index_col=0)
meta = pd.read_csv(f"{BASE}/data/retina_sample_metadata_clean.csv")
# keep AMD vs normal only, retina
ret = meta[meta['group'].isin(['AMD','Normal'])].copy()
ret_genes = [g for g in PALM if g in expr.index]
sub = expr.loc[ret_genes, ret['gsm']].astype(float)
# z-score per gene
z = (sub.sub(sub.mean(axis=1), axis=0)).div(sub.std(axis=1)+1e-9, axis=0)
order = ret.sort_values('group')['gsm'].tolist()
ord_grp = ret.set_index('gsm').loc[order,'group'].astype(str)
zmat = z[order]
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(zmat.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
# color bar
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label('Row z-score', fontsize=8)
# group colors on top
palette = {'AMD':AMD_C,'Normal':NORM_C}
for i,c in enumerate(ord_grp):
    ax.add_patch(mpatches.Rectangle((i-0.5, zmat.shape[0]-0.5), 1, 1, color=palette[c], transform=ax.get_xaxis_transform(), clip_on=False, zorder=5))
ax.set_ylim(-0.5, zmat.shape[0]-0.4)
ax.set_yticks(range(len(ret_genes))); ax.set_yticklabels(ret_genes, fontsize=6.5)
ax.set_xlim(-0.5, len(order)-0.5)
n_amd = sum(1 for x in ord_grp if x=='AMD'); n_norm = len(order)-n_amd
ax.set_xlabel(f'Samples ({n_norm} Normal | {n_amd} AMD)', fontsize=9)
# top legend
lan = mpatches.Patch(color=NORM_C, label='Normal (n=%d)'%n_norm)
laa = mpatches.Patch(color=AMD_C, label='AMD (n=%d)'%n_amd)
ax.legend(handles=[laa,lan], loc='upper left', bbox_to_anchor=(-0.04,-0.12), ncol=2, frameon=False, fontsize=8)
ax.set_xticks([])
ax.set_title('Expression of 29 detectable palmitoylation genes\nacross AMD vs Normal retina (row-wise z-score)', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/Figure2_heatmap_palmitoylation_revised.png", dpi=300)
plt.savefig(f"{FIG}/Figure2_heatmap_palmitoylation_revised.pdf")
plt.close()
print("Figure2 done, detected genes:", len(ret_genes))

# ============ Figure 3: PPI network (STRING v12, score>=0.4) ============
import networkx as nx
edges = pd.read_csv(f"{TAB}/rev_ppi_string_400.csv")
hub = pd.read_csv(f"{TAB}/rev_ppi_hub_genes.csv")
G = nx.Graph()
for _,r in edges.iterrows():
    G.add_edge(r['protein1'], r['protein2'], weight=r['score'])
deg_map = dict(hub.set_index('gene')['degree'])
node_size = [deg_map.get(n, 10)*20 for n in G.nodes]
# label only top hubs (degree >= 13) to avoid overlap
label_genes = {n for n,d in deg_map.items() if d >= 13}
pos = nx.spring_layout(G, seed=7, k=1.1, iterations=200)
fig, ax = plt.subplots(figsize=(7.5, 7))
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.28, width=0.6, edge_color='#AAB7B8')
nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
    node_color=[deg_map.get(n,10) for n in G.nodes], cmap='YlOrRd', edgecolors='#444', linewidths=0.4,
    vmin=min(deg_map.values()), vmax=max(deg_map.values()))
# labels positioned slightly outward
labels = {n: n for n in G.nodes if n in label_genes}
labpos = {n: (p[0], p[1]+0.045) for n,p in pos.items()}
nx.draw_networkx_labels(G, labpos, ax=ax, labels=labels, font_size=7.5, font_color='#111111', font_weight='bold')
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(min(deg_map.values()), max(deg_map.values())))
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Degree', fontsize=8)
ax.set_title('Protein\u2013protein interaction network of palmitoylation genes\n(STRING v12, combined score \u2265 0.4; labels = hub genes, degree \u2265 13)', fontsize=9)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{FIG}/Figure3_ppi_palmitoylation_revised.png", dpi=300)
plt.savefig(f"{FIG}/Figure3_ppi_palmitoylation_revised.pdf")
plt.close()
print("Figure3 done")

# ============ Figure 4: Immune signature scores ============
isc = pd.read_csv(f"{TAB}/rev_immune_signature_scores.csv")
isc = isc.sort_values('p')
fig, ax = plt.subplots(figsize=(7.5, 4.8))
bars = ax.barh(isc['signature'], isc['diff'], color=[AMD_C if d>0 else NORM_C for d in isc['diff']], alpha=0.85)
for i,(_,r) in enumerate(isc.iterrows()):
    ax.errorbar(r['diff'], i, xerr=[[r['diff']-r['ci_lo']],[r['ci_hi']-r['diff']]], fmt='none', ecolor='#444', capsize=1.5, lw=0.7)
ax.set_xlabel('Mean difference (AMD \u2212 Normal), signature score')
ax.set_yticks(range(len(isc))); ax.set_yticklabels(isc['signature'], fontsize=8)
# significance annotation for complement
comp = isc[isc['signature']=='Complement'].iloc[0]
ax.text(comp['ci_hi']+0.03, list(isc['signature']).index('Complement'), 'p=0.010\nq=0.116 (n.s.)', fontsize=6.5, va='center', color='#C0392B')
ax.axvline(0, color='#444', lw=0.6)
ax.set_title('Signature enrichment scores in AMD vs Normal retina\n(mean z-score; error bars = bootstrap 95% CI)', fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Figure4_immune_signature_revised.png", dpi=300)
plt.savefig(f"{FIG}/Figure4_immune_signature_revised.pdf")
plt.close()
print("Figure4 done")

# ============ Figure 5: ROC nested CV donor-grouped (TRUE pooled ROC) ============
# Recompute nested-CV pooled ROC from raw expression to produce an accurate figure.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
expr5 = expr.loc[ret_genes, [g for g in ret['gsm'] if g in expr.columns]].astype(float).T
meta5 = ret.set_index('gsm').reindex(expr5.index)
y5 = (meta5['group']=='AMD').astype(int).values
don5 = meta5['patient identifier'].values
X5 = expr5.values
# donor-grouped outer CV
gkf = GroupKFold(n_splits=5)
pool_fpr, pool_tpr = [], []
aucs = []
all_y, all_p = [], []
for tr_idx, te_idx in gkf.split(X5, y5, groups=don5):
    sc = StandardScaler().fit(X5[tr_idx])
    Xtr, Xte = sc.transform(X5[tr_idx]), sc.transform(X5[te_idx])
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    best_a, best_sc = None, -1
    for a in np.logspace(-3, 0.5, 30):
        m = LogisticRegression(penalty='l1', C=1.0/a, solver='liblinear', max_iter=5000, random_state=42)
        s = cross_val_score(m, Xtr, y5[tr_idx], cv=inner, scoring='roc_auc').mean()
        if s > best_sc:
            best_sc = s; best_a = a
    mf = LogisticRegression(penalty='l1', C=1.0/best_a, solver='liblinear', max_iter=5000, random_state=42)
    mf.fit(Xtr, y5[tr_idx])
    p = mf.predict_proba(Xte)[:,1]
    all_y.extend(y5[te_idx]); all_p.extend(p)
    aucs.append(roc_auc_score(y5[te_idx], p))
all_y = np.array(all_y); all_p = np.array(all_p)
pool_auc = roc_auc_score(all_y, all_p)
fpr, tpr, _ = roc_curve(all_y, all_p)
fig, ax = plt.subplots(figsize=(5.5, 5.2))
ax.plot(fpr, tpr, color=AMD_C, lw=2.3, label=f'LASSO signature (pooled AUC = {pool_auc:.3f})')
ax.fill_between(fpr, tpr, 0, color=AMD_C, alpha=0.08)
ax.plot([0,1],[0,1], ls='--', color='#999999', lw=0.9, label='Chance (AUC = 0.5)')
ax.set_xlabel('False positive rate (1 \u2212 specificity)')
ax.set_ylabel('True positive rate (sensitivity)')
ax.set_title(f'ROC of LASSO palmitoylation signature\nNested CV, donor-grouped folds (pooled AUC = {pool_auc:.3f})', fontsize=8.5)
ax.legend(frameon=False, loc='lower right', fontsize=8)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/Figure5_roc_nested_donor_revised.png", dpi=300)
plt.savefig(f"{FIG}/Figure5_roc_nested_donor_revised.pdf")
plt.close()
print("Figure5 done (true pooled nested-CV ROC, AUC=", round(pool_auc,3), ")")

# ============ Figure 6: Subtype barplot ============
subtypes = {}
for nm,key in [('dry_amd','DRY'),('ga','GA'),('cnv','CNV')]:
    p=f"{TAB}/palm_degs_{nm}_vs_normal.csv"
    if os.path.exists(p):
        d=pd.read_csv(p)
        pcol = 'p' if 'p' in d.columns else 'p_value'
        subtypes[key]=(d[pcol]<0.05).sum()
labels = list(subtypes.keys()); counts = list(subtypes.values())
colors = ['#8E44AD','#16A085','#E67E22']
fig, ax = plt.subplots(figsize=(5.8, 4.6))
bars = ax.bar(labels, counts, color=colors, alpha=0.85, width=0.55)
for b,c in zip(bars, counts):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, str(int(c)), ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Nominal DEGs (p < 0.05)')
ax.set_xlabel('AMD subtype')
ax.set_title('Nominal differentially expressed palmitoylation genes by subtype\n(no gene survived FDR correction)', fontsize=9)
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, max(counts)+3)
plt.tight_layout()
plt.savefig(f"{FIG}/Figure6_subtype_revised.png", dpi=300)
plt.savefig(f"{FIG}/Figure6_subtype_revised.pdf")
plt.close()
print("Figure6 done")
print("ALL MAIN FIGURES REBUILT")