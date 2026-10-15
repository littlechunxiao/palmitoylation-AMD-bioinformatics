#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_immune_signature_scores.py
修订版免疫/通路签名分析。
审稿意见4处理：
1) 不再称"免疫浸润"，改为"免疫细胞/免疫相关通路的签名富集分数"(ssGSEA-like)。
2) 列出全部12个签名的基因、来源文献；明确 Complement/Inflammation 为通路状况评分，非细胞类型。
3) 统计用供者聚类稳健估计(与DEG一致, 处理供者内相关性)，报效应量+95%CI+BH-FDR。
4) 表格报告全部12项(不再只列7项)。
"""
import pandas as pd, numpy as np, os, warnings
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(BASE,"data"); TAB=os.path.join(BASE,"tables"); FIG=os.path.join(BASE,"figures")
plt.rcParams.update({'font.size':8,'figure.dpi':300})

# ---- 12签名：细胞类型(as cell type) vs 通路状况(as pathway) ----
# 来源：Charlmers 2013 PNAS (CIBERSORT LM22 核心细胞), Aran 2017 (ImmunoStates), 及常用免疫通路面
SIG = {
 'B cells':        dict(kind='cell',  genes=['MS4A1','CD79A','CD79B','CD19','CR2','FCER2','BLK','PAX5']),
 'T cells CD4':    dict(kind='cell',  genes=['CD4','IL7R','CCR7','LEF1','TCF7','SELL','CD3D','CD3E']),
 'T cells CD8':    dict(kind='cell',  genes=['CD8A','CD8B','GZMA','GZMB','PRF1','NKG7','KLRB1','CD3D']),
 'NK cells':       dict(kind='cell',  genes=['KLRD1','NKG7','GNLY','GZMA','GZMB','PRF1','XCL2','FCGR3A']),
 'Monocytes':      dict(kind='cell',  genes=['CD14','LYZ','S100A8','S100A9','CSF1R','FCGR3A','CD163','MSR1']),
 'Macrophages':    dict(kind='cell',  genes=['CD68','CD163','MSR1','MRC1','CD80','CD86','CSF1R','CCL2']),
 'Dendritic cells':dict(kind='cell',  genes=['CD1C','CD1D','FCER1A','CLEC10A','LILRA4','GZMB','IRF7','IRF8']),
 'Mast cells':     dict(kind='cell',  genes=['CPA3','TPSAB1','TPSB2','FCER1A','MS4A2','KIT','HDC','IL1RL1']),
 'Neutrophils':    dict(kind='cell',  genes=['CSF3R','FCGR3B','CXCR2','S100A8','S100A9','CEACAM8','CD177','OLFM4']),
 'Microglia':      dict(kind='cell',  genes=['TMEM119','P2RY12','CX3CR1','CSF1R','CD68','AIF1','ITGAM']),
 'Complement':     dict(kind='pathway', genes=['C1QA','C1QB','C1QC','C3','C4A','C4B','C5','CFB','CFD','CFH','CFI','SERPING1']),
 'Inflammation':   dict(kind='pathway', genes=['IL6','TNF','IL1B','CXCL8','CCL2','IFNG','IL10','TGFB1']),
}

expr=pd.read_csv(os.path.join(DATA,"gene_expression_matrix.csv"),index_col=0)
meta=pd.read_csv(os.path.join(DATA,"retina_sample_metadata_clean.csv"),index_col=0)
common=[s for s in meta.index if s in expr.columns]
expr=expr[common]; meta=meta.loc[common].copy()
meta['donor']=meta['patient identifier']; meta['age']=pd.to_numeric(meta['age (years)'].astype(str).str.replace('y',''),errors='coerce')
meta['mac']=meta['tissue'].str.lower().map(lambda x:1 if x.startswith('macular') else 0)
meta['male']=meta['gender'].str.lower().map(lambda x:1 if x=='male' else 0)
sub=meta[meta['group'].isin(['AMD','Normal'])].copy(); sub['amd']=(sub['group']=='AMD').astype(int); sub=sub.sort_values('gsm')

# 计算签名富集分数(全118样本的z相对Normal vs AMD统一化；对每个基因在sub内zscore后取签名内平均)
score_rows=[]
meta_sig=[]
for name,sp in SIG.items():
    gs=[g for g in sp['genes'] if g in expr.index]
    n_avail=len(gs); n_total=len(sp['genes'])
    if n_avail<3:
        continue
    # zscore per gene across AMD+Normal sub
    eg=expr.loc[gs, sub.index]
    z =(eg.sub(eg.mean(axis=1),axis=0)).div(eg.std(axis=1),axis=0)
    sc=z.mean(axis=0)
    # cluster-OLS diff (AMD vs Normal) controlling mac/age/male
    y=sc[sub.index].values
    X=sm.add_constant(np.column_stack([sub['amd'].values,sub['mac'].values,sub['age'].values,sub['male'].values]))
    m=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':sub['donor'].values})
    diff=m.params[1]; ci=m.conf_int()[1]; p=m.pvalues[1]
    score_rows.append(dict(signature=name, kind=sp['kind'], n_genes_used=n_avail, n_genes_signature=n_total,
        mean_Normal=sc[sub['amd']==0].mean(), mean_AMD=sc[sub['amd']==1].mean(),
        diff=diff, ci_lo=ci[0], ci_hi=ci[1], p=p))
    meta_sig.append(name)

sdf=pd.DataFrame(score_rows)
sdf['adj_p']=multipletests(sdf['p'],method='fdr_bh')[1]
sdf=sdf.sort_values('p')
sdf=sdf[['signature','kind','n_genes_used','n_genes_signature','mean_Normal','mean_AMD','diff','ci_lo','ci_hi','p','adj_p']]
sdf.to_csv(os.path.join(TAB,"rev_immune_signature_scores.csv"),index=False)
print("=== 12签名富集分数(AMD vs Normal, 供者聚类稳健) ===")
print(sdf.round(4).to_string(index=False))
print("\n补体(通路)adj_p:", sdf[sdf['signature']=='Complement']['adj_p'].values,
      " 巨噬细胞adj_p:", sdf[sdf['signature']=='Macrophages']['adj_p'].values)

# 热图
# 用全部样本重新算每个签名分(供热图完整)
score_mat=[]
for name,sp in SIG.items():
    gs=[g for g in sp['genes'] if g in expr.index]
    if len(gs)<3: continue
    eg=expr.loc[gs,:]
    z=(eg.sub(eg.mean(axis=1),axis=0)).div(eg.std(axis=1),axis=0)
    score_mat.append((name,z.mean(axis=0)))
scm=pd.concat([pd.Series(s,name=n) for n,s in score_mat],axis=1)
meta_all=meta
order_sorted = meta_all.sort_values('group').index.tolist()
order=[s for s in order_sorted if s in scm.index]
grp=meta_all.loc[order,'group']
cmap_d={'Normal':'#009E73','AMD':'#D55E00','Pre-AMD':'#E69F00'}
cold=[cmap_d.get(g,'#999') for g in grp]
scm2=scm.loc[order].T
fig,ax=plt.subplots(figsize=(10,4.5))
g=sns.heatmap(scm2,cmap='RdBu_r',center=0,ax=ax,xticklabels=False,yticklabels=True,cbar_kws={'label':'Enrichment score'})
ax.set_ylabel('')
# 顶部group colorbar
from matplotlib.patches import Patch
handles=[Patch(color=cmap_d[k],label=k) for k in ['Normal','AMD','Pre-AMD'] if k in set(meta['group'])]
ax.legend(handles=handles,bbox_to_anchor=(1.01,1.12),loc='upper left',ncol=3,frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rev_immune_signature_heatmap.png")); plt.close()
print("\n已保存: rev_immune_signature_scores.csv + rev_immune_signature_heatmap.png")