#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rev_nested_lasso_donor_groupedCV.py
供者分组嵌套CV的严谨版：外层按供者分组(GroupKFold)，内层调参也按供者分组(GroupKFold)，
彻底消除 sample-level 内层折叠导致的同源泄漏。置信区间采用以供者(donor)为单位的 bootstrap 重采样。
覆盖旧输出，供投稿包 Table3/Figure5 更新。
"""
import pandas as pd, numpy as np, os, warnings, sys
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
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
print(f"建模特征(可检测棕榈酰化基因): {len(feat)} | 样本 AMD={int(y.sum())} Normal={int(len(y)-y.sum())} | 供者={len(np.unique(donors))}", flush=True)

gkf = GroupKFold(n_splits=5)
aucs=[]; orders=[]; logits=[]; truths=[]; test_donors=[]; all_feat_counts=[]

for fold,(tr_idx,te_idx) in enumerate(gkf.split(X, y, groups=donors)):
    Xtr,ytr,don_tr = X[tr_idx],y[tr_idx],donors[tr_idx]
    Xte,yte,don_te = X[te_idx],y[te_idx],donors[te_idx]
    assert not (set(don_tr) & set(don_te)), f"外层泄漏 fold{fold}"
    scaler = StandardScaler().fit(Xtr); Xtr_s = scaler.transform(Xtr)
    # ---- 内层按供者分组调参 (GroupKFold)，消除同源泄漏 ----
    inner = GroupKFold(n_splits=4)
    best_a=None; best_score=-1
    for a in np.logspace(-3,0.5,30):
        lasso = LogisticRegression(penalty='l1', C=1.0/a, solver='liblinear', max_iter=5000, random_state=42)
        # 内层评分：用供者分组 CV
        sc_vals=[]
        for itr,ival in inner.split(Xtr_s, ytr, groups=don_tr):
            sc_vals.append(roc_auc_score(ytr[ival], lasso.fit(Xtr_s[itr],ytr[itr]).predict_proba(Xtr_s[ival])[:,1]) if len(np.unique(ytr[ival]))>1 else np.nan)
        sc_vals=[v for v in sc_vals if not np.isnan(v)]
        sc=np.mean(sc_vals) if sc_vals else -1
        if sc>best_score: best_score=sc; best_a=a
    model = LogisticRegression(penalty='l1', C=1.0/best_a, solver='liblinear', max_iter=5000, random_state=42)
    model.fit(Xtr_s, ytr)
    sel = np.where(np.abs(model.coef_[0])>1e-6)[0]
    all_feat_counts.append(int(len(sel)))
    Xte_s = scaler.transform(Xte)
    prob = model.predict_proba(Xte_s)[:,1]
    auc = roc_auc_score(yte, prob)
    aucs.append(auc); truths.extend(yte); logits.extend(prob); test_donors.extend(don_te.tolist())
    fpr, tpr, thr = roc_curve(yte, prob); youden=tpr-fpr; best_thr = thr[np.argmax(youden)]
    orders.append(dict(fold=fold+1, n_test=len(yte), n_amd_test=int(yte.sum()), n_sel_genes=len(sel),
                       best_alpha=best_a, auc=auc, best_threshold=float(best_thr)))
    sel_genes=[feat[i] for i in sel]
    print(f"Fold{fold+1}: n_test={len(yte)} (AMD={int(yte.sum())}) 泄漏=0 选中={len(sel)}({','.join(sel_genes)}) alpha={best_a:.4f} testAUC={auc:.3f}", flush=True)

truths=np.array(truths); logits=np.array(logits); test_donors=np.array(test_donors)
auc_mean=float(np.mean(aucs)); auc_sd=float(np.std(aucs,ddof=1)) if len(aucs)>1 else 0.0
se_auc=auc_sd/np.sqrt(len(aucs)); ci_mean=(auc_mean-1.96*se_auc, auc_mean+1.96*se_auc)
pool_auc=float(roc_auc_score(truths, logits))
fpr,tpr,thr=roc_curve(truths, logits); youden=tpr-fpr; opt_thr=float(thr[np.argmax(youden)])
pred=(logits>=opt_thr).astype(int)
acc=accuracy_score(truths,pred); prec=precision_score(truths,pred); rec=recall_score(truths,pred); f1=f1_score(truths,pred)

# ---- 以供者为重采样单位的 bootstrap CI for pooled AUC ----
rng=np.random.default_rng(2026)
uniq=np.unique(test_donors)
boot_pool=[]
for b in range(2000):
    bs=rng.choice(uniq, size=len(uniq), replace=True)
    idx=[i for d in bs for i in np.where(test_donors==d)[0]]
    if len(np.unique(truths[idx]))>1:
        boot_pool.append(roc_auc_score(truths[idx], logits[idx]))
boot_ci=(float(np.percentile(boot_pool,2.5)), float(np.percentile(boot_pool,97.5))) if boot_pool else (pool_auc,pool_auc)

print("\n=== 供者分组嵌套CV(内层GroupKFold) 结果 ===")
print(f"各折AUC: {np.round(aucs,3)} | 选中特征数/折: {all_feat_counts}")
print(f"平均AUC(外折): {auc_mean:.3f} ± {auc_sd:.3f} (95%CI {ci_mean[0]:.3f}-{ci_mean[1]:.3f})")
print(f"合并AUC: {pool_auc:.3f} (供者bootstrap 95%CI {boot_ci[0]:.3f}-{boot_ci[1]:.3f})")
print(f"最优阈值(Youden): {opt_thr:.3f} | acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

summary=dict(n_features_total=len(feat), cv_folds=5, group_by='donor', inner='GroupKFold(donor)',
             fold_aucs=','.join(f"{a:.3f}" for a in aucs),
             mean_auc=auc_mean, sd_auc=auc_sd, ci_lo=ci_mean[0], ci_hi=ci_mean[1],
             pooled_auc=pool_auc, pooled_ci_lo=boot_ci[0], pooled_ci_hi=boot_ci[1], ci_method='donor-bootstrap(n=2000)',
             optimal_threshold=opt_thr, accuracy=acc, precision=prec, recall=rec, f1=f1,
             n_selected_genes_per_fold=','.join(map(str,all_feat_counts)),
             note="Nested CV; outer AND inner loops donor-grouped (GroupKFold); donor-level bootstrap CI")
pd.DataFrame([summary]).T.to_csv(os.path.join(TAB,"rev_nested_lasso_donor_summary.csv"))
pd.DataFrame(orders).to_csv(os.path.join(TAB,"rev_nested_lasso_donor_perfold.csv"), index=False)

fig,ax=plt.subplots(figsize=(5,5))
ax.plot(fpr,tpr,label=f'Pooled AUC={pool_auc:.3f} (95% CI {boot_ci[0]:.2f}-{boot_ci[1]:.2f})',lw=1.8,color='#2166ac')
ax.plot([0,1],[0,1],'--',color='grey',lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Donor-grouped nested CV: palmitoylation classifier')
ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(os.path.join(FIG,"rev_roc_nested_donor.png")); plt.close()
print("\n已保存 groupedCV 结果(summary/perfold/ROC图)")