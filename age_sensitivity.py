#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审稿意见6：年龄重叠范围敏感性分析。
主分析(all ages) vs 重叠年龄区(>=63岁，排除年轻normal) 通路分×12签名关联对比"""
import pandas as pd, numpy as np, os, warnings
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(BASE,"data"); TAB=os.path.join(BASE,"tables")

SIG = {
 'B cells':['MS4A1','CD79A','CD79B','CD19','CR2','FCER2','BLK','PAX5'],
 'CD4 T cells':['CD4','IL7R','CCR7','LEF1','TCF7','SELL','CD3D','CD3E'],
 'CD8 T cells':['CD8A','CD8B','GZMA','GZMB','PRF1','NKG7','KLRB1','CD3D'],
 'NK cells':['KLRD1','NKG7','GNLY','GZMA','GZMB','PRF1','XCL2','FCGR3A'],
 'Monocytes':['CD14','LYZ','S100A8','S100A9','CSF1R','FCGR3A','CD163','MSR1'],
 'Macrophages':['CD68','CD163','MSR1','MRC1','CD80','CD86','CSF1R','CCL2'],
 'Dendritic cells':['CD1C','CD1D','FCER1A','CLEC10A','LILRA4','GZMB','IRF7','IRF8'],
 'Mast cells':['CPA3','TPSAB1','TPSB2','FCER1A','MS4A2','KIT','HDC','IL1RL1'],
 'Neutrophils':['CSF3R','FCGR3B','CXCR2','S100A8','S100A9','CEACAM8','CD177','OLFM4'],
 'Microglia':['TMEM119','P2RY12','CX3CR1','CSF1R','CD68','AIF1','ITGAM'],
 'Complement':['C1QA','C1QB','C1QC','C3','C4A','C4B','C5','CFB','CFD','CFH','CFI','SERPING1'],
 'Inflammation':['IL6','TNF','IL1B','CXCL8','CCL2','IFNG','IL10','TGFB1'],
}
PALM33 = ['ZDHHC1','ZDHHC2','ZDHHC3','ZDHHC4','ZDHHC5','ZDHHC6','ZDHHC7','ZDHHC8','ZDHHC9',
          'ZDHHC11','ZDHHC12','ZDHHC13','ZDHHC14','ZDHHC15','ZDHHC16','ZDHHC17','ZDHHC18',
          'ZDHHC19','ZDHHC20','ZDHHC21','ZDHHC22','ZDHHC23','ZDHHC24',
          'PPT1','PPT2','LYPLA1','LYPLA2','ABHD10','ABHD12','BST2','LYPLAL1','RAB7A','RAB7B']

expr=pd.read_csv(os.path.join(DATA,"gene_expression_matrix.csv"),index_col=0)
meta=pd.read_csv(os.path.join(DATA,"retina_sample_metadata_clean.csv"),index_col=0)
common=[s for s in meta.index if s in expr.columns]
expr=expr[common]; meta=meta.loc[common].copy()
meta['donor']=meta['patient identifier']
meta['age']=pd.to_numeric(meta['age_num'] if 'age_num' in meta.columns else meta['age (years)'].astype(str).str.replace('y','',regex=False),errors='coerce')
meta['mac']=meta['tissue'].str.lower().map(lambda x:1 if str(x).startswith('macular') else 0)
meta['male']=meta['gender'].str.lower().map(lambda x:1 if str(x)=='male' else 0)

def sig_score(sub):
    s={}
    for name,gs in SIG.items():
        g2=[g for g in gs if g in expr.index]
        if len(g2)<3: s[name]=pd.Series(np.nan,index=sub.index); continue
        eg=expr.loc[g2,sub.index]
        z=(eg.sub(eg.mean(axis=1),axis=0)).div(eg.std(axis=1),axis=0)
        s[name]=z.mean(axis=0)
    return pd.DataFrame(s)

def run(sub,label):
    sub=sub.sort_index()
    sub['amd']=(sub['group']=='AMD').astype(int)
    donor=sub['donor'].values
    gs=[g for g in PALM33 if g in expr.index]
    eg=expr.loc[gs,sub.index]
    z=(eg.sub(eg.mean(axis=1),axis=0)).div(eg.std(axis=1),axis=0)
    palm_pathway=z.mean(axis=0)
    sig=sig_score(sub)
    rows=[]
    for name in sig.columns:
        y=sig.loc[sub.index,name]; yz=(y-y.mean())/y.std()
        X=sm.add_constant(np.column_stack([palm_pathway.values,sub['mac'].values,sub['age'].values,sub['male'].values,sub['amd'].values]))
        m=sm.OLS(yz,X).fit(cov_type='cluster',cov_kwds={'groups':donor})
        b=m.params.iloc[1]; p=m.pvalues.iloc[1]
        rows.append(dict(signature=name,beta=b,p=p))
    res=pd.DataFrame(rows)
    res['fdr']=multipletests(res['p'],method='fdr_bh')[1]
    res=res.sort_values('p')
    print(f"===== {label} (n={len(sub)}, donors={sub['donor'].nunique()}) =====")
    print(res.round(4).to_string(index=False)); print()
    return res

# 主分析(AMD+Normal全年龄)
sub=meta[meta['group'].isin(['AMD','Normal'])].copy()
run(sub,"主分析 all ages")
# 敏感性: 重叠年龄区 >=63岁（排除young normal）
sub2=meta[(meta['group'].isin(['AMD','Normal'])) & (meta['age']>=63)].copy()
run(sub2,"敏感性 age>=63 (重叠区)")
