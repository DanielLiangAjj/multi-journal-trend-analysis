import pickle, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
e=pickle.load(open("data/keyword_embeddings.pkl","rb"))
labels=np.array(e["labels"]); emb=np.asarray(e["embeddings"],dtype=np.float32)
kc=pd.read_csv("data/keywords_categorized.csv")
cat=dict(zip(kc["keyword"], kc["label"]))
lab2idx={l:i for i,l in enumerate(labels)}
def subset(domain):
    keep=[lab2idx[k] for k,c in cat.items() if c in ((domain,"both")) and k in lab2idx]
    return emb[np.array(sorted(set(keep)))]
GRID=[2,5,10,20,30,50,75,100,200]
for domain in ("methodology","health"):
    X=subset(domain)
    print(f"\n[{domain}] n_keywords={len(X)}")
    print(f"  {'K':>5} {'silhouette':>12}")
    for k in GRID:
        if k>=len(X): continue
        km=KMeans(n_clusters=k, n_init=3, random_state=42).fit(X)
        s=silhouette_score(X, km.labels_, sample_size=5000, random_state=42)
        print(f"  {k:>5} {s:>12.4f}")
