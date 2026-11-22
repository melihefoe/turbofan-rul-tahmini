"""app_logic.py — Streamlit arayüzünün iş mantığı (arayüzden bağımsız, test edilebilir).

Okuma → doğrulama → tahmin → risk sınıflandırma adımlarını içerir.
Model, Bölüm 4'te kaydedilen tam pipeline'dır: ham sensör kaydı → kayan özellikler → ölçekleme → LightGBM.
"""
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rul_utils import COLS, FEATURES_14, INDEX_COLS, load_cmapss

BASE = Path(__file__).parent
DATA = BASE / "data"

SENSOR_INFO = {
    "s1": ("T2", "Fan girişi sıcaklık"), "s2": ("T24", "LPC çıkışı sıcaklık"), "s3": ("T30", "HPC çıkışı sıcaklık"),
    "s4": ("T50", "LPT çıkışı sıcaklık"), "s5": ("P2", "Fan girişi basınç"), "s6": ("P15", "Bypass kanalı basınç"),
    "s7": ("P30", "HPC çıkışı toplam basınç"), "s8": ("Nf", "Fiziksel fan hızı"), "s9": ("Nc", "Fiziksel çekirdek hızı"),
    "s10": ("epr", "Motor basınç oranı"), "s11": ("Ps30", "HPC çıkışı statik basınç"), "s12": ("phi", "Yakıt akışı / Ps30"),
    "s13": ("NRf", "Düzeltilmiş fan hızı"), "s14": ("NRc", "Düzeltilmiş çekirdek hızı"), "s15": ("BPR", "Bypass oranı"),
    "s16": ("farB", "Yakıt-hava oranı"), "s17": ("htBleed", "Tahliye entalpisi"), "s18": ("Nf_dmd", "İstenen fan hızı"),
    "s19": ("PCNfR_dmd", "İstenen düzeltilmiş fan hızı"), "s20": ("W31", "HPT soğutma tahliyesi"), "s21": ("W32", "LPT soğutma tahliyesi"),
}

MIN_CYCLES_ERROR = 10          # bundan kısa geçmişle tahmin yapılmaz
MAX_ENGINES = 500
MAX_ROWS = 200_000
RANGE_TOL_STD = 4.0            # eğitim aralığının bu kadar std dışı → uyarı


# --------------------------------------------------------------------------------------
# Yardımcı dosyalar
# --------------------------------------------------------------------------------------
def load_meta():
    meta = json.load(open(DATA / "final_model_meta.json", encoding="utf-8"))
    cfg = json.load(open(DATA / "project_config.json", encoding="utf-8"))
    return meta, cfg


def load_train_ranges():
    return json.load(open(DATA / "train_ranges.json", encoding="utf-8"))


def load_demo_test():
    """Resmi test seti (100 motor) + gerçek RUL değerleri."""
    test = load_cmapss(DATA / "test_FD001.txt")
    rul = pd.read_csv(DATA / "RUL_FD001.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    return test, pd.Series(rul, index=np.arange(1, len(rul) + 1), name="RUL_true")


# --------------------------------------------------------------------------------------
# Okuma ve sütun eşleme
# --------------------------------------------------------------------------------------
_ALIASES = {
    "unit": ["unit", "unit_number", "unit_nr", "engine", "engine_id", "engine_no", "motor", "motor_id", "motor_no", "id"],
    "cycle": ["cycle", "cycles", "time", "time_in_cycles", "time_cycles", "dongu", "döngü", "dongu_no"],
}
_SYMBOL_TO_S = {sym.lower(): s for s, (sym, _) in SENSOR_INFO.items()}


def _normalize_col(c):
    c = str(c).strip().lower().replace(" ", "_").replace("-", "_")
    for key, names in _ALIASES.items():
        if c in names:
            return key
    m = re.fullmatch(r"(?:s|sensor|sensor_|s_|sensör|sensör_)(\d{1,2})", c)
    if m and 1 <= int(m.group(1)) <= 21:
        return f"s{int(m.group(1))}"
    if c in _SYMBOL_TO_S:
        return _SYMBOL_TO_S[c]
    m = re.fullmatch(r"(?:op|setting|op_setting|operational_setting)_?(\d)", c)
    if m and 1 <= int(m.group(1)) <= 3:
        return f"op{int(m.group(1))}"
    return c


def read_uploaded(file_bytes: bytes, filename: str):
    """Yüklenen dosyayı okur. Desteklenen biçimler:
    1) NASA orijinal biçimi: başlıksız, boşlukla ayrılmış 26 sütun (.txt)
    2) Başlıklı CSV / TXT (virgül, noktalı virgül veya sekme ile ayrılmış)
    Dönüş: (DataFrame veya None, mesaj listesi)"""
    msgs = []
    text = file_bytes.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return None, [("error", "Dosya boş.")]
    first = text.strip().splitlines()[0]
    tokens = re.split(r"[\s,;]+", first.strip())
    looks_numeric = all(re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", t, flags=re.I) for t in tokens if t)

    try:
        if looks_numeric:
            raw = pd.read_csv(io.StringIO(text), sep=r"[\s,;]+", header=None, engine="python")
            raw = raw.dropna(axis=1, how="all")
            if raw.shape[1] == 26:
                raw.columns = COLS
                msgs.append(("info", "NASA C-MAPSS orijinal biçimi algılandı (başlıksız, 26 sütun)."))
            else:
                return None, [("error", f"Başlık satırı olmayan dosyada 26 sütun bekleniyordu (NASA biçimi), {raw.shape[1]} sütun bulundu. "
                                        "Başlıklı bir CSV yükleyin ya da NASA biçimini kullanın.")]
        else:
            sep = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first and "," not in first else ",")
            raw = pd.read_csv(io.StringIO(text), sep=sep)
            raw.columns = [_normalize_col(c) for c in raw.columns]
            msgs.append(("info", f"Başlıklı tablo algılandı ({raw.shape[1]} sütun, ayırıcı: '{sep if sep != chr(9) else 'sekme'}')."))
    except Exception as e:  # pragma: no cover - pandas hata mesajı
        return None, [("error", f"Dosya okunamadı: {e}")]
    return raw, msgs


# --------------------------------------------------------------------------------------
# Doğrulama
# --------------------------------------------------------------------------------------
def validate(raw: pd.DataFrame, train_ranges=None, window=50):
    """Girdiyi modele uygun hale getirir ve sorunları raporlar.
    Dönüş: (temiz DataFrame veya None, [(seviye, mesaj), ...]) — seviye: error / warning / info"""
    issues = []
    df = raw.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    if "unit" not in df.columns:
        df.insert(0, "unit", 1)
        issues.append(("warning", "Motor numarası (unit) sütunu yok → tüm satırlar tek bir motor (1) kabul edildi."))
    if "cycle" not in df.columns:
        df.insert(1, "cycle", df.groupby("unit").cumcount() + 1)
        issues.append(("warning", "Döngü (cycle) sütunu yok → satır sırası döngü sırası kabul edildi."))

    missing = [s for s in FEATURES_14 if s not in df.columns]
    if missing:
        sym = ", ".join(f"{s} ({SENSOR_INFO[s][0]})" for s in missing)
        issues.append(("error", f"Modelin ihtiyaç duyduğu {len(missing)} sensör sütunu eksik: {sym}. "
                                "Sütun adları s2…s21, sensor_2… veya T24, T30… biçiminde olabilir."))
        return None, issues

    if len(df) > MAX_ROWS:
        issues.append(("error", f"Dosya çok büyük ({len(df):,} satır). En fazla {MAX_ROWS:,} satır desteklenir."))
        return None, issues

    need = INDEX_COLS + FEATURES_14
    bad_rows = []
    for c in need:
        before = df[c].isna()
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
        newly_bad = df[c].isna() & ~before
        bad_rows += [(i, c) for i in df.index[newly_bad]]
    if bad_rows:
        rows = sorted({i + 2 for i, _ in bad_rows})[:10]       # +2: başlık satırı + 1 tabanlı numara
        cols = sorted({c for _, c in bad_rows})
        issues.append(("error", f"Sayısal olmayan değerler bulundu: sütunlar {cols}, satırlar {rows}{' …' if len(bad_rows) > 10 else ''}. "
                                "Bu hücreleri düzeltip tekrar yükleyin."))
        return None, issues

    if df[["unit", "cycle"]].isna().any().any():
        issues.append(("error", "Motor numarası veya döngü sütununda boş hücre var."))
        return None, issues
    if (df["cycle"] <= 0).any() or (df["cycle"] % 1 != 0).any():
        issues.append(("error", "Döngü değerleri pozitif tam sayı olmalı."))
        return None, issues
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df[FEATURES_14] = df[FEATURES_14].astype(float)

    if df["unit"].nunique() > MAX_ENGINES:
        issues.append(("error", f"En fazla {MAX_ENGINES} motor desteklenir (dosyada {df['unit'].nunique()})."))
        return None, issues

    was_unsorted = not df.groupby("unit", sort=False)["cycle"].apply(lambda c: c.is_monotonic_increasing).all()

    # Eksik sensör değerleri: motor içinde ileri/geri doldurma
    na = df[FEATURES_14].isna()
    if na.any().any():
        n_cells = int(na.sum().sum())
        frac_by_unit = na.any(axis=1).groupby(df["unit"]).mean()
        too_many = frac_by_unit[frac_by_unit > 0.2].index.tolist()
        if too_many:
            issues.append(("error", f"Şu motorlarda satırların %20'sinden fazlasında eksik sensör değeri var: {too_many[:10]}. Güvenilir tahmin yapılamaz."))
            return None, issues
        df = df.sort_values(["unit", "cycle"])
        df[FEATURES_14] = df.groupby("unit")[FEATURES_14].transform(lambda s: s.ffill().bfill())
        issues.append(("warning", f"{n_cells} boş sensör hücresi, aynı motorun komşu döngülerindeki değerlerle dolduruldu."))

    dup = df.duplicated(subset=["unit", "cycle"], keep="last")
    if dup.any():
        df = df[~dup]
        issues.append(("warning", f"{int(dup.sum())} tekrar eden (motor, döngü) satırı bulundu → son kayıt tutuldu."))

    if was_unsorted:
        issues.append(("warning", "Döngüler sıralı değildi → motor ve döngüye göre sıralandı."))
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    gaps = df.groupby("unit")["cycle"].apply(lambda c: int((c.diff().dropna() > 1).sum()))
    if (gaps > 0).any():
        issues.append(("warning", f"Döngü atlaması olan motorlar: {gaps[gaps > 0].index.tolist()[:10]}. Kayan pencere özellikleri mevcut kayıtlarla hesaplanır."))

    n = df.groupby("unit")["cycle"].size()
    too_short = n[n < MIN_CYCLES_ERROR].index.tolist()
    if too_short:
        issues.append(("warning", f"{MIN_CYCLES_ERROR} döngüden kısa kaydı olan motorlar tahmin dışı bırakıldı: {too_short[:10]}."))
        df = df[~df["unit"].isin(too_short)].reset_index(drop=True)
        if df.empty:
            issues.append(("error", f"Tahmin için her motorda en az {MIN_CYCLES_ERROR} döngü gerekli."))
            return None, issues
    short = n[(n >= MIN_CYCLES_ERROR) & (n < window)].index.tolist()
    if short:
        issues.append(("warning", f"{len(short)} motorun geçmişi {window} döngüden kısa ({short[:10]}{' …' if len(short) > 10 else ''}) → "
                                  "bu motorlarda tahmin güveni düşük (test setinde 31 döngülük geçmişte hata ~%23 arttı)."))

    if train_ranges:
        out = []
        for s in FEATURES_14:
            r = train_ranges[s]
            lo, hi = r["min"] - RANGE_TOL_STD * r["std"], r["max"] + RANGE_TOL_STD * r["std"]
            bad_units = df.loc[(df[s] < lo) | (df[s] > hi), "unit"].unique().tolist()
            if bad_units:
                out.append(f"{s} ({SENSOR_INFO[s][0]}) → motor {bad_units[:5]}")
        if out:
            issues.append(("warning", "Eğitim verisinin çok dışında sensör değerleri var (sensör arızası, birim hatası veya farklı çalışma koşulu olabilir). "
                                      "Bu motorlarda tahmin güvenilir değildir: " + "; ".join(out)))

    issues.append(("info", f"{df['unit'].nunique()} motor, {len(df):,} döngü kaydı modele hazır."))
    return df[INDEX_COLS + FEATURES_14], issues


# --------------------------------------------------------------------------------------
# Tahmin ve risk
# --------------------------------------------------------------------------------------
def predict_all_cycles(model, df: pd.DataFrame, cap=125):
    """Her döngü için, o ana kadarki geçmişle yapılan tahmin (model yalnızca geçmişe bakar).
    Son döngünün tahmini = motorun güncel RUL tahmini."""
    p = model.predict(df[INDEX_COLS + FEATURES_14])
    return np.clip(p, 0, cap)


def risk_level(rul, crit=30, plan=75):
    if rul <= crit:
        return "Kritik"
    if rul <= plan:
        return "Bakım planla"
    return "Normal"


RISK_STYLE = {"Kritik": ("🔴", "#d62728"), "Bakım planla": ("🟠", "#ff7f0e"), "Normal": ("🟢", "#2ca02c")}


def confidence(n_cycles, window=50, out_of_range=False):
    if out_of_range:
        return "Düşük (aralık dışı sensör)"
    if n_cycles < 31:
        return "Düşük (çok kısa geçmiş)"
    if n_cycles < window:
        return "Orta (kısa geçmiş)"
    return "Yüksek"


def summarize_engines(model, df: pd.DataFrame, crit=30, plan=75, window=50, cap=125, train_ranges=None):
    """Motor başına güncel RUL tahmini, risk ve güven tablosu."""
    pred = predict_all_cycles(model, df, cap)
    d = df.assign(RUL_pred=pred)
    last = d.groupby("unit").tail(1).set_index("unit")
    n = d.groupby("unit").size()
    oor = pd.Series(False, index=last.index)
    if train_ranges:
        for s in FEATURES_14:
            r = train_ranges[s]
            lo, hi = r["min"] - RANGE_TOL_STD * r["std"], r["max"] + RANGE_TOL_STD * r["std"]
            oor |= d.groupby("unit")[s].apply(lambda v: ((v < lo) | (v > hi)).any())
    out = pd.DataFrame({
        "motor": last.index,
        "son_döngü": last["cycle"].astype(int).to_numpy(),
        "kayıtlı_döngü": n.reindex(last.index).to_numpy(),
        "tahmini_RUL": last["RUL_pred"].round(1).to_numpy(),
    })
    out["risk"] = [risk_level(r, crit, plan) for r in out["tahmini_RUL"]]
    out["güven"] = [confidence(k, window, bool(o)) for k, o in zip(out["kayıtlı_döngü"], oor.reindex(last.index))]
    out["not"] = np.where(out["tahmini_RUL"] >= cap - 10, f"Sağlıklı bölge: model {cap} döngü üzerini ayırt etmez", "")
    return out.reset_index(drop=True), d


def global_importance(model, feature_names):
    """Modelin kendi özellik önemi (LightGBM: gain; yoksa split sayısı)."""
    reg = model.regressor_
    mdl = reg.named_steps["model"]
    imp = None
    try:
        imp = mdl.booster_.feature_importance(importance_type="gain")      # LightGBM
    except Exception:
        imp = getattr(mdl, "feature_importances_", None)
    if imp is None:
        return None
    s = pd.Series(np.asarray(imp, dtype=float), index=feature_names)
    return (s / s.sum() * 100).sort_values(ascending=False)


def feature_names(model):
    return list(model.regressor_.named_steps["features"].get_feature_names_out())


def to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8-sig")
