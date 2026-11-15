# 🧪 Veri Bilimi Mini Projeler

Veri bilimi dersinde öğrendiğim konuları küçük ve çalıştırılabilir projelere döktüğüm koleksiyon. Her klasör bağımsızdır, kendi grafiklerini `cikti/` içine üretir.

| # | Proje | Konu | Temel Sonuç |
|---|---|---|---|
| 1 | [Diyabet Regresyon](01_diyabet_regresyon) | Keşifsel analiz, doğrusal regresyon, Ridge | R² ≈ 0.45, hedefle en ilişkili: BMI |
| 2 | [Müşteri Segmentasyonu](02_musteri_segmentasyonu) | K-Means, silhouette, PCA | En iyi k = 3 (silhouette 0.61) |
| 3 | [Kanser Sınıflandırma](03_kanser_siniflandirma) | Sınıflandırma, çapraz doğrulama, recall | Doğruluk ≈ %98, kötü huylu recall ≈ %96 (CV) |

## 1) Diyabet Regresyon
442 hastanın 10 özelliğinden hastalığın bir yıl sonraki ilerlemesini tahmin eder. Korelasyon matrisi çıkarılır, doğrusal regresyon ile Ridge karşılaştırılır. Veri küçük ve gürültülü olduğu için R² ≈ 0.45 ile sınırlı kalır; bu durum gerçek hayatta her problemin yüksek skora çıkmadığını gösterir.

## 2) Müşteri Segmentasyonu
Yaş, yıllık harcama ve aylık ziyaret sayısına göre müşterileri gruplar. Küme sayısı elle değil **silhouette skoruyla** seçilir. Çıkan segmentler: genç/orta harcamalı sık ziyaretçiler, yüksek harcamalı seyrek müşteriler, düşük harcamalı müşteriler. Veri sentetiktir.

## 3) Kanser Sınıflandırma
Wisconsin meme kanseri verisiyle sınıflandırma yapılır. Burada önemli nokta metriğin seçimidir: kötü huylu vakayı kaçırmak en pahalı hata olduğu için **kötü huylu sınıf recall'u** ölçülür. Bu proje bir tanı aracı değil, ders çalışmasıdır.

## 🚀 Çalıştırma
```bash
pip install -r requirements.txt
python 01_diyabet_regresyon/calistir.py
python 02_musteri_segmentasyonu/calistir.py
python 03_kanser_siniflandirma/calistir.py
```

## 🧰 Teknolojiler
Python · pandas · NumPy · scikit-learn · matplotlib

Veri setleri (diyabet, meme kanseri) scikit-learn ile birlikte gelir; ek indirme gerekmez.
