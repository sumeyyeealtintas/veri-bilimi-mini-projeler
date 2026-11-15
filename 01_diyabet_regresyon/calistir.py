"""Diyabet veri seti: keşifsel analiz + doğrusal regresyon ve Ridge karşılaştırması."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CIKTI = Path(__file__).parent / "cikti"
CIKTI.mkdir(exist_ok=True)

veri = load_diabetes(as_frame=True)
df = veri.frame  # 10 özellik (standartlaştırılmış) + hedef: 1 yıl sonraki hastalık ilerlemesi

print("Boyut:", df.shape)
print("Eksik değer:", int(df.isna().sum().sum()))

# 1) Korelasyon ısı haritası
kor = df.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(kor, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(kor)), kor.columns, rotation=60)
ax.set_yticks(range(len(kor)), kor.columns)
fig.colorbar(im)
ax.set_title("Korelasyon Matrisi")
fig.tight_layout()
fig.savefig(CIKTI / "korelasyon.png", dpi=140)
plt.close(fig)
print("\nHedefle en ilişkili özellikler:")
print(kor["target"].drop("target").abs().sort_values(ascending=False).head(3).round(3))

# 2) Modeller
X, y = df.drop(columns="target"), df["target"]
X_e, X_t, y_e, y_t = train_test_split(X, y, test_size=0.2, random_state=42)
sonuc = {}
for ad, m in {"Doğrusal Regresyon": LinearRegression(), "Ridge (alpha=0.1)": Ridge(alpha=0.1)}.items():
    m.fit(X_e, y_e)
    t = m.predict(X_t)
    sonuc[ad] = {"R2": r2_score(y_t, t), "MAE": mean_absolute_error(y_t, t)}
    print(f"{ad:20s} R2={sonuc[ad]['R2']:.3f}  MAE={sonuc[ad]['MAE']:.1f}")

# 3) Gerçek - tahmin grafiği
en_iyi = LinearRegression().fit(X_e, y_e)
t = en_iyi.predict(X_t)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_t, t, alpha=0.6)
lim = [min(y_t.min(), t.min()), max(y_t.max(), t.max())]
ax.plot(lim, lim, "r--")
ax.set(xlabel="Gerçek", ylabel="Tahmin", title="Gerçek vs Tahmin")
fig.tight_layout()
fig.savefig(CIKTI / "gercek_tahmin.png", dpi=140)
plt.close(fig)

pd.DataFrame(sonuc).T.round(3).to_csv(CIKTI / "sonuclar.csv")
