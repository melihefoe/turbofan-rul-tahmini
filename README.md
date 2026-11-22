# Turbofan Motorlarında Kalan Faydalı Ömür (RUL) Tahmini

BTK Makine Öğrenmesi — uçtan uca proje. NASA C-MAPSS FD001 verisiyle turbofan motorlarının arızaya kadar kaç uçuş döngüsü daha çalışabileceğini tahmin eden bir regresyon modeli ve bu modeli kullanan bir Streamlit karar destek uygulaması.

| | |
|---|---|
| **Canlı uygulama** | [[[Streamlit Cloud bağlantısı] ](https://turbofan-rul-tahmini.streamlit.app](https://turbofan-rul-tahmini-hukza2hamawbzvvnck7yxy.streamlit.app/))|
| **Teknik rapor** | `rapor/BTK_RUL_Teknik_Rapor.pdf` |
| **Hazırlayan** | Melih EFE |

## Sonuçlar (resmi test seti, 100 motor)

| Model | RMSE | MAE | R² | NASA skoru |
|---|---|---|---|---|
| Dummy (ortalama) | 43.07 | 35.90 | −0.07 | 33 629 |
| Baseline: Linear Regression | 21.90 | 17.61 | 0.72 | 1 314 |
| Bölüm 3 adayı: HistGradientBoosting (w=30) | 14.63 | 11.53 | 0.88 | 310 |
| **Final: LightGBM (w=50)** | **13.36** | **10.20** | **0.90** | **275** |

- CV tahmini 14.10 ± 0.57 → test 13.36 (%95 bootstrap GA 11.35–15.35)
- Arızaya yakın motorlarda (RUL ≤ 50) RMSE 6.02; kritik motorların %88'i yakalandı
- Satır bazlı KFold hatayı %38 iyimser gösteriyor → tüm deneyler motor bazlı **GroupKFold** ile

## Depo yapısı

```
├── notebooks/                    # 01 → 06 sırasıyla çalıştırılır (Google Colab)
│   ├── 01_RUL_Problem_Veri_EDA_Kalite.ipynb
│   ├── 02_RUL_Preprocessing_GroupKFold_Baseline.ipynb
│   ├── 03_RUL_Model_Karsilastirma_Aday_Secimi.ipynb
│   ├── 04_RUL_Tuning_Guclu_Rakipler.ipynb
│   ├── 05_RUL_Final_Test_Degerlendirme.ipynb
│   ├── 06_Streamlit_Colab_Calistirma.ipynb
│   └── rul_utils.py              # ortak fonksiyonlar + RollingFeatures dönüştürücüsü
├── streamlit_app/                # canlı uygulama (Streamlit Cloud ana dosyası: streamlit_app/app.py)
│   ├── app.py, app_logic.py, rul_utils.py
│   ├── final_model.joblib        # final LightGBM pipeline'ı (0.8 MB)
│   ├── data/                     # test verisi, test sonuçları, eğitim sensör aralıkları
│   ├── ornek_veriler/            # arayüzü denemek için 5 örnek dosya
│   └── requirements.txt
├── veri/                         # NASA C-MAPSS FD001 ham verisi
│   ├── train_FD001.txt, test_FD001.txt, RUL_FD001.txt
├── sonuclar/                     # Colab'da üretilen gerçek deney çıktıları
│   ├── grid_hgb.csv                       # Grid Search
│   ├── results_stage10_11_trials.csv      # tüm Optuna denemeleri (HGB + 4 rakip)
│   ├── results_stage11_summary.csv        # güçlü rakip özeti
│   ├── reevaluation.csv                   # bağımsız yeniden değerlendirme (fold bazlı)
│   ├── results_stage12_*.csv              # resmi test: metrikler, tahminler, sağlamlık
│   └── final_model_meta.json              # final model parametreleri ve sürümler
├── rapor/BTK_RUL_Teknik_Rapor.pdf
├── requirements_notebooks.txt
└── README.md
```

## Notebook'ları çalıştırma (Google Colab)

1. Veri dosyaları `veri/` klasöründedir (kaynak: Kaggle "NASA Turbofan Jet Engine Data Set" / NASA Open Data Portal, C-MAPSS).
2. Google Drive'da `BTK_RUL` klasörü açın; `veri/` içindeki üç dosyayı ve `notebooks/rul_utils.py`'yi buraya koyun.
3. Notebook'ları 01 → 06 sırasıyla Colab'da açıp çalıştırın. Her notebook Drive'ı bağlar ve kararları `project_config.json` üzerinden bir sonrakine aktarır.
4. 04 numaralı notebook (tuning) ücretsiz Colab'da yaklaşık 1,5 saat sürer; kesilirse `checkpoints/` klasöründen kaldığı yerden devam eder.
5. Test dosyaları yalnızca 05 numaralı notebook'ta açılır.

Yerelde çalıştırmak için: `pip install -r requirements_notebooks.txt` ve notebook'lardaki `BASE` yolunu kendi klasörünüzle değiştirin.

## Uygulamayı yerelde çalıştırma

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Veri kaynağı ve lisans

A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08, 2008. NASA Ames Prognostics Center of Excellence tarafından kamuya açık yayımlanmış bir simülasyon verisidir ve kişisel veri içermez. [Lisans ifadesini Kaggle/NASA sayfasından teyit edip buraya yazın.]

## Sınırlılıklar

Model yalnızca FD001 koşulları (tek çalışma koşulu, HPC bozulması) için geçerlidir; 125 döngünün üzerindeki kalan ömürleri ayırt edemez; veri simülasyondur. Uygulama bir **karar destek** aracıdır, bakım kararının yerine geçmez.

## Yapay zekâ kullanım beyanı

Bu projede kodların, deney tasarımının, grafiklerin, raporun ve sunumun hazırlanmasında Claude (Anthropic) yapay zekâ asistanından destek alınmıştır. Notebook'lar Google Colab'da çalıştırılmıştır; raporlanan tüm sonuçlar gerçek deney çıktılarıdır.
