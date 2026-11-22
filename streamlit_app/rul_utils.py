"""rul_utils.py — NASA C-MAPSS RUL projesi için ortak yardımcılar.
Notebook'lar ve Streamlit uygulaması aynı kodu kullanır (tekrarlanabilirlik)."""
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS
FEATURES_14 = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
               "s13", "s14", "s15", "s17", "s20", "s21"]


def load_cmapss(path):
    """Boşlukla ayrılmış, başlıksız C-MAPSS dosyasını okur."""
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def add_rul(df):
    """Eğitim verisi için gerçek (kırpılmamış) RUL: motorun son döngüsü - mevcut döngü."""
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return df


def _rolling_slope(x, window):
    """Geçmiş `window` döngüdeki doğrusal eğim (yalnızca geçmişe bakar)."""
    s = pd.Series(np.asarray(x, dtype=float))
    t = pd.Series(np.arange(len(s), dtype=float))
    cov = s.rolling(window, min_periods=2).cov(t)
    var = t.rolling(window, min_periods=2).var()
    return (cov / var).fillna(0.0).to_numpy()


class RollingFeatures(BaseEstimator, TransformerMixin):
    """Motor bazında, yalnızca GEÇMİŞ döngüleri kullanan kayan pencere özellikleri.

    - Girdi: 'unit', 'cycle' ve sensör sütunlarını içeren DataFrame (motor içinde döngü sırasına göre).
    - Çıktı: ham sensörler (+ isteğe bağlı) kayan ortalama, standart sapma ve eğim.
    - 'unit' ve 'cycle' çıktıya ALINMAZ (özellik değil, yalnızca gruplama/sıra bilgisi).
    - fit() hiçbir şey öğrenmez (stateless): bir satırın özellikleri yalnızca kendi motorunun
      o ana kadarki kayıtlarından hesaplanır → başka motorlardan veya gelecekten bilgi sızmaz.
    """

    def __init__(self, sensors=tuple(FEATURES_14), window=10, stats=("mean", "std", "slope"), keep_raw=True):
        self.sensors = sensors
        self.window = window
        self.stats = stats
        self.keep_raw = keep_raw

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if not X.groupby("unit")["cycle"].apply(lambda c: c.is_monotonic_increasing).all():
            raise ValueError("Her motorun satırları döngü sırasına göre sıralı olmalı.")
        sensors, w = list(self.sensors), int(self.window)
        g = X.groupby("unit", sort=False)
        parts = []
        if self.keep_raw:
            parts.append(X[sensors].reset_index(drop=True))
        if "mean" in self.stats:
            m = g[sensors].transform(lambda c: c.rolling(w, min_periods=1).mean())
            parts.append(m.add_suffix(f"_mean{w}").reset_index(drop=True))
        if "std" in self.stats:
            sd = g[sensors].transform(lambda c: c.rolling(w, min_periods=2).std()).fillna(0.0)
            parts.append(sd.add_suffix(f"_std{w}").reset_index(drop=True))
        if "slope" in self.stats:
            sl = g[sensors].transform(lambda c: _rolling_slope(c, w))
            parts.append(sl.add_suffix(f"_slope{w}").reset_index(drop=True))
        out = pd.concat(parts, axis=1)
        out.index = X.index
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.sensors) if self.keep_raw else []
        for st in ("mean", "std", "slope"):
            if st in self.stats:
                names += [f"{s}_{st}{self.window}" for s in self.sensors]
        return np.array(names)


def nasa_score(y_true, y_pred):
    """PHM08 skor fonksiyonu. d = tahmin - gerçek.
    Geç tahmin (d > 0, ömrü fazla tahmin etmek) daha ağır cezalandırılır."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true, y_pred, prefix=""):
    return {
        f"{prefix}RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}R2": float(r2_score(y_true, y_pred)),
        f"{prefix}NASA": nasa_score(y_true, y_pred),
    }


def test_like_rows(df, idx, n_cuts=10, min_cycle=31, max_rul=150, seed=42):
    """Validation motorlarından resmi test protokolünü taklit eden satırlar seçer:
    her motor için rastgele `n_cuts` kesim noktası; kesim anındaki GERÇEK RUL ile değerlendirilir.
    - min_cycle: kayan pencerelerin dolması için gereken minimum geçmiş.
    - max_rul: bakım karar ufku. Kesim noktaları gerçek RUL <= max_rul olan döngülerden seçilir
      (EDA: bozulma ömrün ikinci yarısında görünür hale geliyor; daha erken döngülerde RUL
      sensörlerden ayırt edilemez ve bakım kararı açısından önemsizdir)."""
    rng = np.random.default_rng(seed)
    sub = df.loc[idx]
    chosen = []
    for _, d in sub.groupby("unit"):
        cand = d.index[(d["cycle"] >= min(min_cycle, d["cycle"].max())) & (d["RUL"] <= max_rul)]
        chosen.extend(rng.choice(cand, size=min(n_cuts, len(cand)), replace=False))
    return np.sort(np.array(chosen))


def cv_evaluate(model, df, cap, cv, feature_cols=None, n_cuts=10, max_rul=150, seed=42, name=None,
                train_metrics=False, model_size=False, return_preds=False):
    """Motor bazlı CV ile değerlendirme.
    - Eğitim hedefi: RUL üst sınırla kırpılır (cap=None ise kırpılmaz).
    - 'clip_' metrikleri: validation motorlarının tüm döngüleri, kırpılmış hedefe göre.
    - 'test_' metrikleri: validation motorlarında rastgele kesim noktaları, GERÇEK RUL'a göre
      ve bakım karar ufku (gerçek RUL <= max_rul) içinde → model ve cap seçiminde birincil ölçüt.
    - train_metrics=True: eğitim fold'undaki RMSE (aşırı öğrenme kontrolü).
    - model_size=True: eğitilmiş modelin bellekteki (pickle) boyutu, MB.
    - return_preds=True: (sonuç_tablosu, kesim_noktası_tahminleri) döndürür (hata analizi için).
    """
    feature_cols = feature_cols or (INDEX_COLS + FEATURES_14)
    X = df[feature_cols]
    y_true = df["RUL"].to_numpy()
    y_fit = y_true if cap is None else np.minimum(y_true, cap)
    rows, preds = [], []
    for k, (tr, va) in enumerate(cv.split(X, y_fit, groups=df["unit"])):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y_fit[tr])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = np.clip(m.predict(X.iloc[va]), 0, None)
        pred_s = time.perf_counter() - t0
        res = {"fold": k + 1, "fit_s": fit_s, "pred_s": pred_s}
        res.update(regression_metrics(y_fit[va], p, "clip_"))
        cut = test_like_rows(df, df.index[va], n_cuts=n_cuts, max_rul=max_rul, seed=seed + k)
        pos = pd.Series(np.arange(len(va)), index=df.index[va]).loc[cut].to_numpy()
        res.update(regression_metrics(y_true[cut], p[pos], "test_"))
        # Arızaya yakın bölge (gerçek RUL <= 50): bakım kararının en kritik olduğu dönem
        near = y_true[cut] <= 50
        res["near50_RMSE"] = float(np.sqrt(np.mean((p[pos][near] - y_true[cut][near]) ** 2))) if near.any() else np.nan
        # Geç tahmin oranı: ömrü 10 döngüden fazla FAZLA tahmin etme (güvenlik açısından tehlikeli)
        res["late10_pct"] = float(np.mean(p[pos] - y_true[cut] > 10) * 100)
        if train_metrics:
            p_tr = np.clip(m.predict(X.iloc[tr]), 0, None)
            res["train_clip_RMSE"] = float(np.sqrt(mean_squared_error(y_fit[tr], p_tr)))
        if model_size:
            import pickle
            res["size_MB"] = len(pickle.dumps(m)) / 1e6
        rows.append(res)
        if return_preds:
            preds.append(pd.DataFrame({"model": name, "fold": k + 1, "unit": df.loc[cut, "unit"].to_numpy(),
                                       "cycle": df.loc[cut, "cycle"].to_numpy(),
                                       "RUL_true": y_true[cut], "RUL_pred": p[pos]}))
    out = pd.DataFrame(rows)
    if name is not None:
        out.insert(0, "model", name)
    if return_preds:
        return out, pd.concat(preds, ignore_index=True)
    return out


def summarize(cv_df, by="model"):
    cols = [c for c in ["test_RMSE", "test_MAE", "test_R2", "test_NASA", "near50_RMSE", "late10_pct",
                        "clip_RMSE", "train_clip_RMSE", "fit_s", "pred_s", "size_MB"] if c in cv_df.columns]
    agg = cv_df.groupby(by)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.sort_values("test_RMSE_mean")


def oof_predictions(model, df, cap, cv, feature_cols=None):
    """Out-of-fold tahminler: her satır, motoru validation'dayken eğitilen modelden tahmin edilir."""
    feature_cols = feature_cols or (INDEX_COLS + FEATURES_14)
    X = df[feature_cols]
    y_fit = df["RUL"].to_numpy() if cap is None else np.minimum(df["RUL"].to_numpy(), cap)
    oof = np.zeros(len(df))
    for tr, va in cv.split(X, y_fit, groups=df["unit"]):
        m = clone(model).fit(X.iloc[tr], y_fit[tr])
        oof[va] = np.clip(m.predict(X.iloc[va]), 0, None)
    return oof
