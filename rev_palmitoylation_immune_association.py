#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_palmitoylation_immune_association.py
审稿意见主要问题3处理：直接检验 棕榈酰化基因/通路表达 与 免疫/补体签名评分 的关联。
设计（供体感知）：
- 棕榈酰化通路分 = 33个可检测棕榈酰化基因的跨样本 z-score 均值（与签名分同套做法）
- 单基因：33个可检测基因逐一与签名分关联
- 模型：供者聚类稳健 OLS；棕榈酰化(基因或通路分) 为预测因子，
  协变量 = AMD状态 + 区域(macular) + 年龄 + 性别  -> 直接回应审稿"说明如何处理疾病状态、区域、年龄、性别"
- BH-FDR 多重校正，明确探索性
- 输出：关联结果表 + 关联热图(通路分×12签名) + 关键散点图
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

# ---- 12签名（与 rev_immune_signature_scores.py 完全一致）----
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

# ---- 33个可检测棕榈酰化基因（来自 rev_palmitoylation_41genes_deg.csv 的 platform_detectable=True）----
PALM33 = ['ZDHHC1','ZDHHC2','ZDHHC3','ZDHHC4','ZDHHC5','ZDHHC6','ZDHHC7','ZDHHC8','ZDHHC9',
          'ZDHHC11','ZDHHC12','ZDHHC13','ZDHHC14','ZDHHC15','ZDHHC16','ZDHHC17','ZDHHC18',
          'ZDHHC19','ZDHHC20','ZDHHC21','ZDHHC22','ZDHHC23','ZDHHC24',
          'PPT1','PPT2','LYPLA1','LYPLA2','ABHD10','ABHD12','BST2','LYPLAL1','RAB7A','RAB7B']

def load_scores():
    """复用签名脚本逻辑：对每个样本算 12 个签名的 z-score-mean 分"""
    expr=pd.read_csv(os.path.join(DATA,"gene_expression_matrix.csv"),index_col=0)
    meta=pd.read_csv(os.path.join(DATA,"retina_sample_metadata_clean.csv"),index_col=0)
    common=[s for s in meta.index if s in expr.columns]
    expr=expr[common]; meta=meta.loc[common].copy()
    meta['donor']=meta['patient identifier']
    meta['age']=pd.to_numeric(meta['age (years)'].astype(str).str.replace('y','',regex=False),errors='coerce')
    meta['mac']=meta['tissue'].str.lower().map(lambda x:1 if str(x).startswith('macular') else 0)
    meta['male']=meta['gender'].str.lower().map(lambda x:1 if str(x)=='male' else 0)
    # AMD vs Normal 子集（与分类器/签名一致；pre-AMD 排除）
    sub=meta[meta['group'].isin(['AMD','Normal'])].copy()
    sub['amd']=(sub['group']=='AMD').astype(int)
    sub=sub.sort_index()
    # 签名分
    sig_score={}
    for name,sp in SIG.items():
        gs=[g for g in sp['genes'] if g in expr.index]
        if len(gs)<3: 
            sig_score[name]=pd.Series(np.nan,index=expr.columns); continue
        eg=expr.loc[gs,:]
        z=(eg.sub(eg.mean(axis=1),axis=0)).div(eg.std(axis=1),axis=0)
        sig_score[name]=z.mean(axis=0)
    sig_df=pd.DataFrame(sig_score)  # samples x 12
    return expr, meta, sub, sig_df

def main():
    expr, meta, sub, sig_df = load_scores()
    donor=sub['donor'].values
    # ---- 棕榈酰化通路分（33基因 z-score 均值，AMD+Normal 子集内 zscore）----
    gs=[g for g in PALM33 if g in expr.index]
    eg=expr.loc[gs, sub.index]
    z=(eg.sub(eg.mean(axis=1),axis=0)).div(eg.std(axis=1),axis=0)
    palm_pathway=z.mean(axis=0)  # sample-level pathway score

    def term(y, cov):
        X=sm.add_constant(np.column_stack([y.values, cov['mac'], cov['age'], cov['male'], cov['amd']]))
        m=sm.OLS(palm_pathway.values, X).fit(cov_type='cluster',cov_kwds={'groups':sub['donor'].values})
        return m

    # ---- 通路分 × 12签名 ----
    rows=[]
    for name in sig_df.columns:
        y=sig_df.loc[sub.index, name]
        yz=(y-y.mean())/y.std()
        X=sm.add_constant(np.column_stack([palm_pathway.values, sub['mac'].values, sub['age'].values, sub['male'].values, sub['amd'].values]))
        m=sm.OLS(yz, X).fit(cov_type='cluster',cov_kwds={'groups':donor})
        b=m.params.iloc[1]; ci=m.conf_int().iloc[1].values; p=m.pvalues.iloc[1]
        rows.append(dict(palmitoylation='Pathway_score', signature=name, kind=SIG[name]['kind'],
                         beta=b, ci_lo=ci[0], ci_hi=ci[1], p=p))
    res=pd.DataFrame(rows)
    res['fdr']=multipletests(res['p'],method='fdr_bh')[1]
    res=res.sort_values('p')
    print("=== 棕榈酰化通路分 × 免疫/补体签名（供体聚类稳健OLS，控制AMD/区域/年龄/性别）===")
    print(res.round(4).to_string(index=False))

    # ---- 单基因 × 关键签名（补体+巨噬细胞+炎症，供体感知）----
    key_sigs=['Complement','Macrophages','Inflammation']
    rows2=[]
    for gene in gs:
        ge=expr.loc[gene, sub.index]
        gz=(ge-ge.mean())/ge.std()
        for name in key_sigs:
            y=sig_df.loc[sub.index, name]; yz=(y-y.mean())/y.std()
            X=sm.add_constant(np.column_stack([gz.values, sub['mac'].values, sub['age'].values, sub['male'].values, sub['amd'].values]))
            m=sm.OLS(yz, X).fit(cov_type='cluster',cov_kwds={'groups':donor})
            b=m.params.iloc[1]; ci=m.conf_int().iloc[1].values; p=m.pvalues.iloc[1]
            rows2.append(dict(gene=gene, signature=name, beta=b, ci_lo=ci[0], ci_hi=ci[1], p=p))
    res2=pd.DataFrame(rows2)
    res2['fdr']=multipletests(res2['p'],method='fdr_bh')[1]
    res2=res2.sort_values('p')
    print("\n=== 33棕榈酰化基因 × (补体/巨噬/炎症) 供体聚类关联（Top15）===")
    print(res2.head(15).round(4).to_string(index=False))

    # ---- 保存 ----
    res.to_csv(os.path.join(TAB,"rev_palm_pathway_immune_association.csv"),index=False)
    res2.to_csv(os.path.join(TAB,"rev_palm_gene_immune_association.csv"),index=False)

    # ---- 热图：通路分 × 12 签名 ----
    pv=res.pivot_table(index='signature',columns='palmitoylation',values='beta')
    fig,ax=plt.subplots(figsize=(4.5,6))
    sns.heatmap(pv,cmap='RdBu_r',center=0,annot=True,fmt='.2f',ax=ax,
                cbar_kws={'label':'Standardized β (palm score)'})
    ax.set_title('Palmitoylation-pathway score × immune/complement signatures')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"rev_palm_immune_assoc_heatmap.png"),dpi=300); plt.savefig(os.path.join(FIG,"rev_palm_immune_assoc_heatmap.pdf")); plt.close()

    # ---- 关键散点图：通路分 vs 补体分数，按AMD着色，供体符号 ----
    df_plot=pd.DataFrame({'palm':palm_pathway.loc[sub.index].values,
                          'Complement':sig_df.loc[sub.index,'Complement'].values,
                          'amd':sub['amd'].values,'gsm':sub.index})
    fig,ax=plt.subplots(figsize=(5,4))
    for lab,s in [('Normal',0),('AMD',1)]:
        d=df_plot[df_plot['amd']==s]
        ax.scatter(d['palm'],d['Complement'],label=lab,s=18,alpha=.7)
    # 供体连接 (配对样本)
    for d in sub['donor'].unique():
        idx=sub.index[sub['donor']==d]
        if len(idx)==2:
            ax.plot([palm_pathway.loc[i] for i in idx],[sig_df.loc[i,'Complement'] for i in idx],'k-',lw=.4,alpha=.4)
    ax.set_xlabel('Palmitoylation-pathway score (z)'); ax.set_ylabel('Complement score (z)')
    ax.legend(frameon=False); ax.set_title('Pathway score vs Complement score (donor-paired)')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"rev_palm_complement_scatter.png"),dpi=300); plt.savefig(os.path.join(FIG,"rev_palm_complement_scatter.pdf")); plt.close()

    print("\n已保存: rev_palm_pathway_immune_association.csv / rev_palm_gene_immune_association.csv")
    print("已保存图: rev_palm_immune_assoc_heatmap.png / rev_palm_complement_scatter.png")

if __name__=="__main__":
    main()