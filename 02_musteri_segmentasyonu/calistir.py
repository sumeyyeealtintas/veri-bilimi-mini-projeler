"""K-Means ile müşteri segmentasyonu: en iyi küme sayısını silhouette ile seçer."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CIKTI = Path(__file__).parent / "cikti"
CIKTI.mkdir(exist_ok=True)

rng = np.random.default_rng(0)
# Üç gizli grup: tasarruflu, orta, yüksek harcamalı
gruplar = [
    (200, [45, 30, 4], [8, 6, 1.5]),    # yaş, yıllık harcama (bin ₺), ziyaret/ay
    (200, [30, 90, 12], [6, 15, 3]),
    (150, [55, 160, 6], [7, 20, 2]),
]
df = pd.concat(
    [pd.DataFrame(rng.normal(m, s, (n, 3)), columns=["yas", "harcama_bin_tl", "ziyaret_ay"])
     for n, m, s in gruplar],
    ignore_index=True,
).clip(lower=1)

Z = StandardScaler().fit_transform(df)

skorlar = {}
for k in range(2, 8):
    etiket = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    skorlar[k] = silhouette_score(Z, etiket)
en_iyi_k = max(skorlar, key=skorlar.get)
print("Silhouette skorları:", {k: round(v, 3) for k, v in skorlar.items()})
print("Seçilen küme sayısı:", en_iyi_k)

df["segment"] = KMeans(n_clusters=en_iyi_k, n_init=10, random_state=0).fit_predict(Z)
profil = df.groupby("segment").agg(
    kisi=("yas", "size"), yas=("yas", "mean"),
    harcama_bin_tl=("harcama_bin_tl", "mean"), ziyaret_ay=("ziyaret_ay", "mean"),
).round(1)
print("\nSegment profilleri:\n", profil)
profil.to_csv(CIKTI / "segment_profilleri.csv")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(list(skorlar), list(skorlar.values()), "o-")
a1.axvline(en_iyi_k, color="r", ls="--")
a1.set(xlabel="Küme sayısı (k)", ylabel="Silhouette", title="Küme Sayısı Seçimi")
p = PCA(2).fit_transform(Z)
a2.scatter(p[:, 0], p[:, 1], c=df["segment"], cmap="viridis", s=12)
a2.set(title="Segmentler (PCA ile 2 boyut)", xlabel="PC1", ylabel="PC2")
fig.tight_layout()
fig.savefig(CIKTI / "segmentler.png", dpi=140)
