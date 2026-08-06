import pandas as pd, numpy as np, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
# corpus articles per year (offset)
yc=pd.Series(0,index=range(2011,2026))
for ch in pd.read_csv("data/pubmed_all_journals_2011_2025_w_keywords_patched.csv",usecols=["year"],chunksize=100000):
    v=ch.year.value_counts(); 
    for y in range(2011,2026): yc[y]+=int(v.get(y,0))
corpus=yc.values.astype(float)
years=np.arange(2011,2026); xc=years-2011
print("corpus/yr:", dict(zip(years,corpus.astype(int))))
def run(domain):
    df=pd.read_csv(f"data/visualizations_k100/{domain}_topic_year_counts.csv")
    df.columns=[c.strip().strip('\"') for c in df.columns]
    yk=[str(y) for y in years]
    res=[]
    for _,r in df.iterrows():
        y=np.array([float(r[k]) for k in yk])
        if y.sum()<10: continue
        X=sm.add_constant(xc)
        try:
            m=sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(corpus)).fit(scale="X2")
            coef=m.params[1]; p=m.pvalues[1]
        except Exception: 
            continue
        res.append((r["topic"], coef, p))
    R=pd.DataFrame(res, columns=["topic","share_logslope","p"])
    rej,padj,_,_=multipletests(R.p, alpha=0.05, method="holm")
    R["p_holm"]=padj; R["sig"]=rej
    npos=int(((R.share_logslope>0)&R.sig).sum()); nneg=int(((R.share_logslope<0)&R.sig).sum())
    print(f"\n[{domain}] n={len(R)} topics tested | Holm-sig SHARE trends: {int(R.sig.sum())} ({100*R.sig.mean():.0f}%) -> {npos} rising-share, {nneg} declining-share")
    print(f"   top rising-share:", list(R[(R.share_logslope>0)&R.sig].nlargest(5,'share_logslope')['topic']))
    print(f"   declining-share (Holm-sig):", list(R[(R.share_logslope<0)&R.sig].nsmallest(8,'share_logslope')['topic']))
    R.to_csv(f"data/visualizations_k100/{domain}_share_trends.csv", index=False)
    return R
for d in ("methodology","health"): run(d)
