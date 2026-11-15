"""Meme kanseri veri seti: sınıflandırma, çapraz doğrulama ve hata türü analizi."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, make_scorer, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CIKTI = Path(__file__).parent / "cikti"
CIKTI.mkdir(exist_ok=True)

veri = load_breast_cancer()
X_e, X_t, y_e, y_t = train_test_split(
    veri.data, veri.target, test_size=0.2, stratify=veri.target, random_state=42
)

modeller = {
    "Lojistik Regresyon": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}
# Veri setinde 0 = malignant (kötü huylu), 1 = benign (iyi huylu).
# Tıbbi problemde kötü huylu vakayı kaçırmak en pahalı hatadır; bu yüzden
# recall'u kötü huylu sınıf (pos_label=0) için ölçüyoruz.
kotu_huylu_recall = make_scorer(recall_score, pos_label=0)
for ad, m in modeller.items():
    cv = cross_val_score(m, X_e, y_e, cv=5, scoring=kotu_huylu_recall)
    print(f"{ad:20s} CV recall (kötü huylu): {cv.mean():.3f} ± {cv.std():.3f}")

model = modeller["Lojistik Regresyon"].fit(X_e, y_e)
print(classification_report(y_t, model.predict(X_t), target_names=veri.target_names))

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_estimator(model, X_t, y_t, display_labels=["Kötü huylu", "İyi huylu"], cmap="Blues", ax=ax)
ax.set_title("Karışıklık Matrisi")
fig.tight_layout()
fig.savefig(CIKTI / "karisiklik_matrisi.png", dpi=140)
