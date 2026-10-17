#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_deg_donor_cluster.py
修订版差异表达分析
策略：AMD vs Normal，全部118样本（63用于AMD/Normal比较），
     用 OLS + 供者(patient)聚类稳健标准误 处理供者内相关性(=limma duplicateCorrelation 的稳健近似)，
     同时纳入组织区域(macular/extramacular)、年龄、性别为协变量。
     报告效应量(Log2FC)、95% CI、原始p、BH-FDR。
     不做重复log2（表达矩阵已在对数尺度）。
补充：对棕榈酰化33个可检测基因，另用 MixedLM(供者随机效应) 交叉验证，两种方法对比。
"""
import pandas as pd, numpy as np, statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
TAB  = os.path.join(BASE, "tables")

expr = pd.read_csv(os.path.join(DATA,"gene_expression_matrix.csv"), index_col=0)
meta = pd.read_csv(os.path.join(DATA,"retina_sample_metadata_clean.csv"), index_col=0)

common = [s for s in meta.index if s in expr.columns]
expr = expr[common]; meta = meta.loc[common].copy()
meta['age'] = pd.to_numeric(meta['age (years)'].astype(str).str.replace('y',''), errors='coerce')
meta['tissue_cat'] = meta['tissue'].str.strip().str.lower().map(lambda x:'mac' if x.startswith('macular') else 'extra')
meta['gender_cat'] = meta['gender'].astype(str).str.lower()
meta['donor'] = meta['patient identifier']
sub = meta[meta['group'].isin(['AMD','Normal'])].copy()
sub['amd'] = (sub['group']=='AMD').astype(int)
sub = sub.sort_values('gsm')
print(f"AMD vs Normal 样本: AMD={ (sub['amd']==1).sum() }, Normal={ (sub['amd']==0).sum() }, 供者={sub['donor'].nunique()}")

X = np.column_stack([
    sub['amd'].values,
    (sub['tissue_cat']=='mac').astype(int).values,
    sub['age'].values,
    (sub['gender_cat']=='male').astype(int).values,
])
Xd = sm.add_constant(X)
clusters = sub['donor'].values

# 所有表达基因
all_genes = expr.index.tolist()
rows=[]
for g in all_genes:
    y = expr.loc[g, sub.index].values
    try:
        m = sm.OLS(y, Xd).fit(cov_type='cluster', cov_kwds={'groups': clusters})
        coef = m.params[1]; se = m.bse[1]; p = m.pvalues[1]
        ci_lo, ci_hi = m.conf_int()[1]
        rows.append(dict(gene=g, log2fc=coef, se=se, ci_lo=ci_lo, ci_hi=ci_hi, p=p,
                         t=m.tvalues[1]))
    except Exception as e:
        rows.append(dict(gene=g, log2fc=np.nan,se=np.nan,ci_lo=np.nan,ci_hi=np.nan,p=np.nan,t=np.nan))

deg = pd.DataFrame(rows).set_index('gene')
# BH-FDR
deg['fdr'] = np.nan
mask = deg['p'].notna()
deg.loc[mask,'fdr'] = multipletests(deg.loc[mask,'p'], method='fdr_bh')[1]
deg = deg.sort_values('p')
deg.to_csv(os.path.join(TAB,"rev_DEG_amd_vs_normal_donorcluster.csv"))
sig = deg[deg['p']<0.05]
print(f"\n全转录组 DEG(原始p<0.05): {len(sig)} 个; FDR<0.05: {(deg['fdr']<0.05).sum()} 个")
print("Top 10 差异基因:")
print(sig.head(10)[['log2fc','ci_lo','ci_hi','p','fdr']].round(4).to_string())

# ---- 棕榈酰化 41 基因核对 ----
PALM = ['ZDHHC1','ZDHHC2','ZDHHC3','ZDHHC4','ZDHHC5','ZDHHC6','ZDHHC7','ZDHHC8','ZDHHC9','ZDHHC11','ZDHHC12','ZDHHC13','ZDHHC14','ZDHHC15','ZDHHC16','ZDHHC17','ZDHHC18','ZDHHC19','ZDHHC20','ZDHHC21','ZDHHC22','ZDHHC23','ZDHHC24','ZDHHC25','PPT1','PPT2','LYPLA1','LYPLA2','ABHD10','ABHD12','ABHD16A','ABHD17A','ABHD17B','ABHD17C','BST2','GCP16','LYPLAL1','RAB7A','RAB7B','SELENOK','SELENOS']
gene_rows=[]
for g in PALM:
    det = g in expr.index
    if det and g in deg.index:
        r = deg.loc[g]
        gene_rows.append(dict(gene=g, platform_detectable=det, log2fc=r['log2fc'], ci_lo=r['ci_lo'], ci_hi=r['ci_hi'], p=r['p'], fdr=r['fdr']))
    else:
        gene_rows.append(dict(gene=g, platform_detectable=det, log2fc=np.nan,ci_lo=np.nan,ci_hi=np.nan,p=np.nan,fdr=np.nan))
palm = pd.DataFrame(gene_rows)
palm.to_csv(os.path.join(TAB,"rev_palmitoylation_41genes_deg.csv"), index=False)
det = palm['platform_detectable'].sum()
print(f"\n41基因中平台可检测: {det} 个 (未覆盖 {41-det} 个)")
print("\n41基因差异表达(可检测部分):")
print(palm[palm['platform_detectable']].round(4).to_string())

# ---- 混合效应交叉验证(MixedLM 供者随机效应) ----
print("\n=== MixedLM 供者随机效应 复核棕榈酰化可检测基因 ===")
ml=[]
for g in palm[palm['platform_detectable']]['gene']:
    tmp = sub.copy(); tmp['y']=expr.loc[g, sub.index].values
    tmp['mac'] = (tmp['tissue_cat']=='mac').astype(int)
    tmp['male'] = (tmp['gender_cat']=='male').astype(int)
    try:
        m = mixedlm("y ~ amd + mac + age + male", tmp, groups=tmp['donor']).fit()
        coef=m.params['amd']; se=m.bse['amd']; p=m.pvalues['amd']
        ml.append(dict(gene=g, log2fc_mixed=coef, p_mixed=p))
    except Exception as e:
        ml.append(dict(gene=g, log2fc_mixed=np.nan, p_mixed=np.nan))
mlf = pd.DataFrame(ml)
palm = palm.merge(mlf, on='gene', how='left')
palm.to_csv(os.path.join(TAB,"rev_palmitoylation_41genes_deg.csv"), index=False)
print(palm[palm['platform_detectable']][['gene','log2fc','p','fdr','log2fc_mixed','p_mixed']].round(4).to_string())
print("\n完成: rev_DEG_amd_vs_normal_donorcluster.csv + rev_palmitoylation_41genes_deg.csv")