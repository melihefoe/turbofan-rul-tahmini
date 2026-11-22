# ✈️ Turbofan Motorlarında Kalan Faydalı Ömür (RUL) Tahmini

NASA C-MAPSS FD001 verisiyle eğitilmiş bir makine öğrenmesi modelinin Streamlit arayüzü. Bir turbofan motorunun
sensör geçmişinden **arızaya kadar kaç uçuş döngüsü kaldığını** tahmin eder ve risk seviyesini gösterir.

| | |
|---|---|
| **Model** | LightGBM (kayan pencere özellikleri: son 50 döngünün ortalaması, std ve eğimi) |
| **Test RMSE** | 13.36 döngü (%95 GA: 11.35 – 15.35) · baseline Linear Regression: 21.90 |
| **Arızaya yakın motorlarda (RUL ≤ 50) RMSE** | 6.02 döngü |
| **Kritik motorları yakalama** | %88 (22/25) |

## Sayfalar
- **🔍 Test motoru analizi:** 100 test motorundan birini seçin; tahmin, gerçek değer ve zaman içindeki simülasyon.
- **📤 Kendi verinle tahmin:** NASA biçiminde veya başlıklı CSV olarak sensör kaydı yükleyin; doğrulama raporu, motor bazında tahmin ve CSV indirme.
- **🧪 Ya şöyle olursa?:** Sensör değerlerini değiştirip tahminin duyarlılığını inceleyin.
- **📊 Model performansı:** Test metrikleri, risk karışıklık matrisi, özellik önemi, literatür karşılaştırması.

`ornek_veriler/` klasöründe arayüzü denemek için hazır dosyalar vardır (geçerli ve kasıtlı olarak hatalı örnekler).

## Yerelde çalıştırma
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dosyalar
| Dosya | Açıklama |
|---|---|
| `app.py` | Streamlit arayüzü |
| `app_logic.py` | Dosya okuma, doğrulama, tahmin ve risk hesapları |
| `rul_utils.py` | Eğitimde kullanılan özellik dönüşümü (`RollingFeatures`) — model yüklenirken gereklidir |
| `final_model.joblib` | Eğitilmiş tam pipeline (özellikler → ölçekleme → LightGBM) |
| `data/` | Test seti, gerçek RUL değerleri, sonuç tabloları ve model bilgileri |

> ⚠️ Bu uygulama bir karar destek aracıdır ve simülasyon verisiyle eğitilmiştir. Bakım kararları yetkili mühendis onayıyla verilmelidir.

**Veri kaynağı:** A. Saxena, K. Goebel, D. Simon, N. Eklund (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM 2008, NASA Ames Prognostics Center of Excellence.
