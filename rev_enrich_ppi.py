#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_enrich_ppi.py
修订版富集 + PPI 分析。审稿意见6处理：明确工具/版本/背景/多重校正；STRING证据+阈值敏感性。
- 富集: Enrichr (gseapy 1.3.1), 基因集 GO_Biological_Process/Molecular_Function/Cellular_Component+KEGG_2021_Human+Reactome_2022
        背景=Enrichr全库, 多重校正=Benjamini-Hochberg(Enrichr输出Adjusted P)
  输入=供者稳健全转录组 p<0.05 的DEG(up/down分方向)
  RA: 棕榈酰化差异基因太少(仅4名义显著)，故富集用于全转录组DEG；棕榈酰化通路主题通过补体通路与单基因讨论体现
- PPI: STRING-DB API v12, 证据=combined_score(默认>0.4), 并做敏感度(0.4/0.7)
"""
import pandas as pd, numpy as np, os, warnings
import gseapy as gp
import requests, json
warnings.filterwarnings('ignore')

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(BASE,"data"); TAB=os.path.join(BASE,"tables"); FIG=os.path.join(BASE,"figures")
os.makedirs(TAB,exist_ok=True)

# ---- 富集输入：修订版全转录组 DEG ----
deg=pd.read_csv(os.path.join(TAB,"rev_DEG_amd_vs_normal_donorcluster.csv"),index_col=0)
sig=deg[deg['p']<0.05]
up=sig[sig['log2fc']>0].index.tolist()
down=sig[sig['log2fc']<0].index.tolist()
print(f"富集输入: p<0.05 DEG共{len(sig)} (上调{len(up)}, 下调{len(down)}); FDR<0.05仅{(deg['fdr']<0.05).sum()}")

ENR_DIR=os.path.join(BASE,"enrichr_results"); os.makedirs(ENR_DIR,exist_ok=True)
def run_enr(genes,desc,gs_name,gs_list):
    if len(genes)<5: 
        print(f"  {desc}/{gs_name}: 基因太少({len(genes)})，跳过")
        return None
    try:
        r=gp.enrichr(gene_list=genes, gene_sets=gs_list, organism='human',
                     outdir=os.path.join(ENR_DIR,desc.replace(' ','_')), no_plot=True, cutoff=0.05)
        df=r.results.copy()
        df.to_csv(os.path.join(TAB,f"rev_enrich_{desc}_{gs_name}.csv"),index=False)
        sig_rows=df[df['Adjusted P-value']<0.05]
        print(f"  {desc}/{gs_name}: {len(df)}项, 显著(adj<0.05){len(sig_rows)}")
        if len(sig_rows):
            print(sig_rows[['Term','P-value','Adjusted P-value','Combined Score']].head(8).round(3).to_string(index=False))
        return df
    except Exception as e:
        print(f"  {desc}/{gs_name} Enrichr失败: {e}")
        return None

set_map={'GO_BP':['GO_Biological_Process_2023'],'GO_MF':['GO_Molecular_Function_2023'],
         'GO_CC':['GO_Cellular_Component_2023'],'KEGG':['KEGG_2021_Human'],'Reactome':['Reactome_2022']}
groups=[('all_degs',sig.index.tolist()),('upregulated',up),('downregulated',down)]
print("\n=== 富集分析(Enrichr, BH校正) ===")
for gname,glist in groups:
    for gs_name,gs_list in set_map.items():
        run_enr(glist,gname,gs_name,gs_list)

# ---- PPI: STRING v12 ----
print("\n=== PPI (STRING-DB v12) ===")
from collections import Counter
def string_net(genes,score):
    url="https://string-db.org/api/json/network"
    params={"identifiers":"%0d".join(genes),"species":9606,
            "required_score":int(score*1000),"network_type":"functional","caller_identity":"AMD_palmitoylation_research"}
    try:
        r=requests.post(url,data=params,timeout=60)
        return r.json()
    except Exception as e:
        print("STRING err",e); return None
# 用41棕榈酰化基因(可检测33)建网络, 两个阈值做敏感度
palm=['ZDHHC1','ZDHHC2','ZDHHC3','ZDHHC4','ZDHHC5','ZDHHC6','ZDHHC7','ZDHHC8','ZDHHC9','ZDHHC11','ZDHHC12','ZDHHC13','ZDHHC14','ZDHHC15','ZDHHC16','ZDHHC17','ZDHHC18','ZDHHC19','ZDHHC20','ZDHHC21','ZDHHC22','ZDHHC23','ZDHHC24','ZDHHC25','PPT1','PPT2','LYPLA1','LYPLA2','ABHD10','ABHD12','ABHD16A','ABHD17A','ABHD17B','ABHD17C','BST2','GCP16','LYPLAL1','RAB7A','RAB7B','SELENOK','SELENOS']
usable=[g for g in palm if g in deg.index]
print(f"PPI用基因(可检测棕榈酰化): {len(usable)}")
for score in [0.4,0.7]:
    data=string_net(usable,score)
    if data:
        edge=[(d['preferredName_A'],d['preferredName_B'],d.get('score',0)) for d in data]
        df=pd.DataFrame(edge,columns=['protein1','protein2','score'])
        df.to_csv(os.path.join(TAB,f"rev_ppi_string_{int(score*1000)}.csv"),index=False)
        deg_count=Counter()
        for a,b,s in edge:
            deg_count[a]+=1; deg_count[b]+=1
        hubs=sorted(deg_count.items(),key=lambda x:-x[1])[:10]
        print(f"  STRING score>={score}: {len(edge)}条边, {len(set([x for e in edge for x in e[:2]]))}节点")
        print("  Top hub:", [f"{g}({n})" for g,n in hubs])
        if score==0.4:
            hubs_df=pd.DataFrame(hubs,columns=['gene','degree'])
            hubs_df.to_csv(os.path.join(TAB,"rev_ppi_hub_genes.csv"),index=False)
print("\n完成: rev_enrich_* + rev_ppi_*")