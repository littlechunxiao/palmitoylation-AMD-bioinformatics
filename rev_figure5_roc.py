#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_figure5_roc.py
正式投稿版 Figure 5：供者分组(donor-grouped)嵌套交叉验证的合并 ROC 曲线。
- 复用 rev_nested_lasso_donor_groupedCV.py 的完全一致 CV 流程（外层/内层均 GroupKFold 按供者分组），
  保证与正文/Table 3 报告的 Pooled AUC = 0.689  95%CI 0.559-0.805 严格一致。
- 输出投稿包替代 0.665 旧图，消除图中/正文数值不一致。
"""
import pandas as pd, numpy as np, os, warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE,"data"); TAB = os.path.join(BASE,"tables"); FIG = os.path.join(BASE,"figures")
plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':11,'figure.dpi':300})

expr = pd.read_csv(os.path.join(DATA,"gene_expression_matrix.csv"), index_col=0)
meta = pd.read_csv(os.path.join(DATA,"retina_sample_metadata_clean.csv"), index_col=0)
common=[s for s in meta.index if s in expr.columns]
expr=expr[common]; meta=meta.loc[common].copy()
meta['donor']=meta['patient identifier']
sub=meta[meta['group'].isin(['AMD','Normal'])].copy()
sub['amd']=(sub['group']=='AMD').astype(int)
sub=sub.sort_values('gsm')

PALM=['ZDHHC1','ZDHHC2','ZDHHC3','ZDHHC4','ZDHHC5','ZDHHC6','ZDHHC7','ZDHHC8','ZDHHC9','ZDHHC11','ZDHHC12','ZDHHC13','ZDHHC14','ZDHHC15','ZDHHC16','ZDHHC17','ZDHHC18','ZDHHC19','ZDHHC20','ZDHHC21','ZDHHC22','ZDHHC23','ZDHHC24','ZDHHC25','PPT1','PPT2','LYPLA1','LYPLA2','ABHD10','ABHD12','ABHD16A','ABHD17A','ABHD17B','ABHD17C','BST2','GCP16','LYPLAL1','RAB7A','RAB7B','SELENOK','SELENOS']
feat=[g for g in PALM if g in expr.index]
X = expr.loc[feat, sub.index].T.values
y = sub['amd'].values
donors = sub['donor'].values

gkf = GroupKFold(n_splits=5)
logits=[]; truths=[]; test_donors=[]; aucs=[]
for fold,(tr_idx,te_idx) in enumerate(gkf.split(X, y, groups=donors)):
    Xtr,ytr,don_tr = X[tr_idx],y[tr_idx],donors[tr_idx]
    Xte,yte,don_te = X[te_idx],y[te_idx],donors[te_idx]
    scaler = StandardScaler().fit(Xtr); Xtr_s = scaler.transform(Xtr)
    inner = GroupKFold(n_splits=4)
    best_a=None; best_score=-1
    for a in np.logspace(-3,0.5,30):
        lasso = LogisticRegression(penalty='l1', C=1.0/a, solver='liblinear', max_iter=5000, random_state=42)
        sc_vals=[]
        for itr,ival in inner.split(Xtr_s, ytr, groups=don_tr):
            sc_vals.append(roc_auc_score(ytr[ival], lasso.fit(Xtr_s[itr],ytr[itr]).predict_proba(Xtr_s[ival])[:,1]) if len(np.unique(ytr[ival]))>1 else np.nan)
        sc_vals=[v for v in sc_vals if not np.isnan(v)]
        sc=np.mean(sc_vals) if sc_vals else -1
        if sc>best_score: best_score=sc; best_a=a
    model = LogisticRegression(penalty='l1', C=1.0/best_a, solver='liblinear', max_iter=5000, random_state=42)
    model.fit(Xtr_s, ytr)
    p = model.predict_proba(scaler.transform(Xte))[:,1]
    logits.extend(p); truths.extend(yte); test_donors.extend(don_te)
    aucs.append(roc_auc_score(yte, p))

truths=np.array(truths); logits=np.array(logits)
pool_auc=float(roc_auc_score(truths, logits))
fpr,tpr,_=roc_curve(truths, logits)

# 供者 bootstrap CI（与权威脚本一致, n=2000, seed=2026）
rng=np.random.default_rng(2026); uniq=np.unique(test_donors); boot=[]
for b in range(2000):
    bs=rng.choice(uniq,size=len(uniq),replace=True)
    idx=[i for d in bs for i in np.where(np.array(test_donors)==d)[0]]
    if len(np.unique(truths[idx]))>1:
        boot.append(roc_auc_score(truths[idx], logits[idx]))
ci_lo,ci_hi=float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))

# 一致性校验：必须与正文/Table 3 严格一致
ref=pd.read_csv(os.path.join(TAB,"rev_nested_lasso_donor_summary.csv"), index_col=0, header=None)
ref_pool=float(ref.iloc[ref.index.get_loc('pooled_auc'),0])
assert abs(pool_auc-ref_pool)<1e-6, f"AUC不一致: 本脚本{pool_auc:.4f} vs 权威{ref_pool:.4f}"
assert abs(ci_lo-float(ref.iloc[ref.index.get_loc('pooled_ci_lo'),0]))<0.005 and abs(ci_hi-float(ref.iloc[ref.index.get_loc('pooled_ci_hi'),0]))<0.005, "CI不一致"
print(f"OK 一致性通过: Pooled AUC={pool_auc:.3f} (95%CI {ci_lo:.3f}-{ci_hi:.3f})")

# ---- 绘制正式投稿版 Figure 5 ----
fig,ax=plt.subplots(figsize=(5.6,5.2))
ax.plot(fpr,tpr,color='#2166ac',lw=2.2,label=f'Palmitoylation LASSO signature\n(pooled AUC = {pool_auc:.3f}, 95% CI {ci_lo:.2f}\u2013{ci_hi:.2f})')
ax.fill_between(fpr,tpr,0,color='#2166ac',alpha=0.08)
ax.plot([0,1],[0,1],ls='--',color='#999999',lw=0.9,label='Chance (AUC = 0.5)')
ax.set_xlabel('False positive rate (1 \u2212 specificity)')
ax.set_ylabel('True positive rate (sensitivity)')
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,loc='lower right',fontsize=8.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG,"Figure5_roc_nested_donor_revised.png"),dpi=300)
plt.savefig(os.path.join(FIG,"Figure5_roc_nested_donor_revised.pdf"))
plt.close()
print("Figure 5 已更新: 输出 Figure5_roc_nested_donor_revised.png/pdf (AUC=0.689)")