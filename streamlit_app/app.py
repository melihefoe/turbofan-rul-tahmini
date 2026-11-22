"""Turbofan Motor Kalan Ömür (RUL) Tahmini — Streamlit arayüzü.

Çalıştırma:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_logic import (DATA, RISK_STYLE, SENSOR_INFO, feature_names, global_importance, load_demo_test,
                       load_meta, load_train_ranges, read_uploaded, risk_level, summarize_engines,
                       to_csv_bytes, validate)
from rul_utils import FEATURES_14, INDEX_COLS

BASE = Path(__file__).parent
st.set_page_config(page_title="Turbofan RUL Tahmini", page_icon="✈️", layout="wide")


# ======================================================================================
# Yükleme (önbellekli)
# ======================================================================================
@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model():
    return joblib.load(BASE / "final_model.joblib")


@st.cache_data(show_spinner=False)
def get_static():
    meta, cfg = load_meta()
    return meta, cfg, load_train_ranges()


@st.cache_data(show_spinner="Test motorları hazırlanıyor...")
def get_demo(_model, window, cap):
    test, rul = load_demo_test()
    summary, per_cycle = summarize_engines(_model, test, window=window, cap=cap)
    return test, rul, summary, per_cycle


try:
    model = get_model()
except Exception as e:
    st.error("Model dosyası yüklenemedi. `final_model.joblib` dosyasının uygulamayla aynı klasörde olduğundan ve "
             "`requirements.txt` içindeki scikit-learn / LightGBM sürümlerinin modelin kaydedildiği sürümlerle aynı "
             f"olduğundan emin olun.\n\nHata: `{e}`")
    st.stop()

meta, cfg, TRAIN_RANGES = get_static()
WINDOW, CAP = int(meta["window"]), int(meta["cap"])
test_df, rul_true, demo_summary, demo_cycles = get_demo(model, WINDOW, CAP)
FEAT_NAMES = feature_names(model)


# ======================================================================================
# Kenar çubuğu
# ======================================================================================
st.sidebar.title("✈️ Turbofan RUL")
PAGES = ["🏠 Genel bakış", "🔍 Test motoru analizi", "📤 Kendi verinle tahmin", "🧪 Ya şöyle olursa?", "📊 Model performansı"]
page = st.sidebar.radio("Sayfa", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk eşikleri")
st.sidebar.caption("Örnek bir iş kuralı; bakım politikanıza göre değiştirebilirsiniz.")
CRIT = st.sidebar.slider("🔴 Kritik: RUL ≤", 5, 60, int(cfg.get("risk_thresholds", {}).get("kritik", 30)), step=5)
PLAN = st.sidebar.slider("🟠 Bakım planla: RUL ≤", CRIT + 5, 120, max(int(cfg.get("risk_thresholds", {}).get("bakim_planla", 75)), CRIT + 5), step=5)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{meta['final_model']}** · pencere {WINDOW} döngü · RUL üst sınırı {CAP}\n\n"
                   f"Test RMSE: **{cfg.get('test_RMSE', float('nan')):.2f}** döngü")


# ======================================================================================
# Ortak görsel bileşenler
# ======================================================================================
def risk_badge(rul):
    lvl = risk_level(rul, CRIT, PLAN)
    emoji, color = RISK_STYLE[lvl]
    return lvl, f"<span style='background:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>{emoji} {lvl}</span>"


def show_prediction_card(rul_pred, n_cycles, true_rul=None):
    lvl, badge = risk_badge(rul_pred)
    c1, c2, c3 = st.columns(3)
    label = f"{rul_pred:.0f} döngü" if rul_pred < CAP - 10 else f"≥ {CAP - 10} döngü"
    c1.metric("Tahmini kalan ömür", label)
    if true_rul is not None:
        c2.metric("Gerçek kalan ömür", f"{true_rul:.0f} döngü", delta=f"hata {rul_pred - true_rul:+.1f}", delta_color="off")
    else:
        c2.metric("Kullanılan geçmiş", f"{n_cycles} döngü")
    c3.markdown(f"**Risk seviyesi**<br><br>{badge}", unsafe_allow_html=True)
    if rul_pred >= CAP - 10:
        st.info(f"Motor sağlıklı bölgede. Model, kalan ömrü {CAP} döngünün üzerinde olan motorları birbirinden ayırt edemez "
                "(bu bölgede sensörlerde bozulma izi henüz yoktur).")
    if n_cycles < WINDOW:
        st.warning(f"Bu motorun yalnızca {n_cycles} döngülük geçmişi var. Model en iyi {WINDOW}+ döngülük geçmişle çalışır; "
                   "tahmin güveni düşük.")
    if lvl == "Kritik":
        st.error("Motor kritik bölgede: bakım planlamasını öne alın. (Karar destek amaçlıdır; nihai karar mühendis onayıyla verilir.)")


def plot_rul_trajectory(cycles, preds, true_end=None, upto=None, title=""):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(cycles, preds, lw=2, label="Tahmini RUL (o ana kadarki geçmişle)")
    if true_end is not None:
        true_curve = true_end + (cycles.max() - cycles)
        ax.plot(cycles, np.minimum(true_curve, CAP), "k--", lw=1, label=f"Gerçek RUL (üst sınır {CAP})")
    ax.axhspan(0, CRIT, color="#d62728", alpha=.08); ax.axhspan(CRIT, PLAN, color="#ff7f0e", alpha=.06)
    if upto is not None:
        ax.axvline(upto, c="gray", ls=":", lw=1)
    ax.set_xlabel("döngü"); ax.set_ylabel("RUL (döngü)"); ax.set_ylim(0, CAP + 10); ax.set_title(title); ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)


def plot_sensors(d, sensors, upto=None):
    if not sensors:
        return
    n = len(sensors)
    cols = min(n, 2); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), squeeze=False)
    for ax, s in zip(axes.flat, sensors):
        ax.plot(d["cycle"], d[s], lw=.8, alpha=.5, label="ham")
        ax.plot(d["cycle"], d[s].rolling(WINDOW, min_periods=1).mean(), lw=2, label=f"{WINDOW} döngü ortalama")
        r = TRAIN_RANGES[s]
        ylim = ax.get_ylim()
        ax.axhspan(r["min"], r["max"], color="green", alpha=.06)
        ax.set_ylim(ylim)                      # bant eksen ölçeğini değiştirmesin
        if upto is not None:
            ax.axvline(upto, c="gray", ls=":", lw=1)
        ax.set_title(f"{s} — {SENSOR_INFO[s][0]} ({SENSOR_INFO[s][1]})", fontsize=9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Yeşil zemin: eğitim verisindeki değer aralığı (grafik sınırları içinde kalan kısmı).")


def sensor_label(s):
    return f"{s} — {SENSOR_INFO[s][0]} ({SENSOR_INFO[s][1]})"


DEFAULT_SENSORS = ["s11", "s4", "s12", "s14"]


# ======================================================================================
# 1) Genel bakış
# ======================================================================================
if page == PAGES[0]:
    st.title("✈️ Turbofan Motorlarında Kalan Faydalı Ömür (RUL) Tahmini")
    st.markdown(
        "Bu uygulama, bir turbofan motorunun uçuş döngüleri boyunca kaydedilen sensör ölçümlerinden **arızaya kadar kaç "
        "döngü daha çalışabileceğini** tahmin eder. Bakım planlama mühendisleri için bir **karar destek** aracıdır."
    )
    c = st.columns(4)
    c[0].metric("Test RMSE", f"{cfg.get('test_RMSE', np.nan):.2f} döngü")
    ci = cfg.get("test_RMSE_CI95", [np.nan, np.nan])
    c[1].metric("%95 güven aralığı", f"{ci[0]:.1f} – {ci[1]:.1f}")
    c[2].metric("Arızaya yakın motorlarda RMSE", f"{cfg.get('test_RMSE_near50', np.nan):.1f} döngü")
    c[3].metric("Kritik motorları yakalama", f"%{cfg.get('test_critical_recall_pct', np.nan):.0f}")

    st.subheader("Nasıl kullanılır?")
    st.markdown(
        f"""
| Sayfa | Ne yapar? |
|---|---|
| **🔍 Test motoru analizi** | NASA test setindeki 100 motordan birini seçin; sensör geçmişi, tahmini kalan ömür ve **gerçek** değer yan yana gösterilir. Kaydırıcıyla motorun ömrü boyunca tahminin nasıl değiştiğini izleyin. |
| **📤 Kendi verinle tahmin** | Bir veya birden fazla motorun sensör kaydını (NASA biçimi veya başlıklı CSV) yükleyin; her motor için kalan ömür, risk ve güven seviyesi hesaplanır, sonuçlar CSV olarak indirilebilir. |
| **🧪 Ya şöyle olursa?** | Bir sensörün son döngülerdeki değerini değiştirip tahminin nasıl etkilendiğini görün. |
| **📊 Model performansı** | Test sonuçları, risk sınıfı karışıklık matrisi, özellik önemi ve literatür karşılaştırması. |
        """
    )
    st.subheader("Model hakkında")
    st.markdown(
        f"""
- **Veri:** NASA C-MAPSS FD001 — 100 eğitim motoru (arızaya kadar tam kayıt), 100 test motoru. 21 sensörden bilgi taşıyan **14**'ü kullanılır.
- **Özellikler:** Her sensör için son **{WINDOW} döngünün** ortalaması, oynaklığı (std) ve eğimi (bozulma hızı). Yalnızca geçmiş döngüler kullanılır.
- **Model:** {meta['final_model']} — 5 farklı algoritma arasından, motor bazlı çapraz doğrulama ve eşit tuning bütçesiyle seçildi.
- **Hedef:** Kalan ömür {CAP} döngüde sınırlandırılmıştır; sağlıklı bölgedeki motorlar "≥ {CAP - 10}" olarak gösterilir.
- **Girdi gereksinimi:** Motor başına en az 10 döngü; **{WINDOW}+ döngü önerilir** (31 döngülük geçmişte test hatası ~%23 arttı).
        """
    )
    st.warning("⚠️ Bu bir karar destek aracıdır ve simülasyon verisiyle eğitilmiştir. Bakım kararları mutlaka yetkili mühendis onayıyla verilmelidir.")


# ======================================================================================
# 2) Test motoru analizi + zaman içinde simülasyon
# ======================================================================================
elif page == PAGES[1]:
    st.title("🔍 Test motoru analizi")
    st.caption("NASA FD001 resmi test seti: her motor ömrünün bilinmeyen bir noktasında kesilmiştir; gerçek kalan ömür ayrıca bilinmektedir.")

    opts = demo_summary.copy()
    opts["gerçek"] = opts["motor"].map(rul_true)
    labels = {int(r.motor): f"Motor {int(r.motor)} — {int(r.kayıtlı_döngü)} döngü kayıt, gerçek RUL {int(r.gerçek)}" for r in opts.itertuples()}
    unit = st.selectbox("Motor seçin", list(labels), index=33, format_func=lambda u: labels[u])

    d = demo_cycles[demo_cycles["unit"] == unit]
    last_cycle = int(d["cycle"].max())
    true_end = float(rul_true[unit])

    st.markdown("##### ⏱️ Zaman içinde simülasyon")
    upto = st.slider("Kaçıncı döngüdeyiz? (model yalnızca bu ana kadarki veriyi görür)", int(d["cycle"].min()), last_cycle, last_cycle)
    cur = d[d["cycle"] <= upto]
    pred_now = float(cur["RUL_pred"].iloc[-1])
    true_now = true_end + (last_cycle - upto)
    show_prediction_card(pred_now, len(cur), true_rul=true_now)

    plot_rul_trajectory(d["cycle"].to_numpy(), d["RUL_pred"].to_numpy(), true_end=true_end, upto=upto,
                        title=f"Motor {unit}: tahmini kalan ömrün zaman içindeki değişimi")
    st.caption("Her noktadaki tahmin, yalnızca o döngüye kadarki sensör verisiyle yapılmıştır. "
               "Siyah kesikli çizgi gerçek kalan ömrü gösterir (model eğitimde 125'te sınırlandığı için aynı sınırla çizilmiştir).")

    st.markdown("##### Sensör geçmişi")
    sensors = st.multiselect("Gösterilecek sensörler", FEATURES_14, default=DEFAULT_SENSORS, format_func=sensor_label)
    plot_sensors(d, sensors, upto=upto)


# ======================================================================================
# 3) Kendi verinle tahmin (dosya yükleme)
# ======================================================================================
elif page == PAGES[2]:
    st.title("📤 Kendi verinle tahmin")
    with st.expander("📄 Dosya biçimi ve şablon", expanded=False):
        st.markdown(
            f"""
**Kabul edilen biçimler**
1. **NASA C-MAPSS orijinal biçimi:** başlık satırı olmayan, boşlukla ayrılmış 26 sütun (`.txt`).
2. **Başlıklı tablo (`.csv` / `.txt`):** virgül, noktalı virgül veya sekme ile ayrılmış; ondalık ayırıcı nokta veya virgül olabilir.
   - Motor numarası: `unit`, `engine`, `motor_no`, `id` …  · Döngü: `cycle`, `time`, `dongu` …
   - Sensörler: `s2`, `sensor_2`, `T24` … biçimlerinden biri. Gerekli 14 sensör: {", ".join(f"`{s}` ({SENSOR_INFO[s][0]})" for s in FEATURES_14)}

**Kurallar:** her motor için en az 10 döngü (önerilen {WINDOW}+). Boş hücreler aynı motorun komşu döngüleriyle doldurulur;
sayısal olmayan değerler hata verir; tekrar eden satırlar ve sıralama otomatik düzeltilir.
            """
        )
        tmpl = pd.DataFrame(columns=["unit", "cycle"] + FEATURES_14)
        st.download_button("⬇️ Boş şablon (CSV)", to_csv_bytes(tmpl), "rul_sablon.csv", "text/csv")

    examples = {"— örnek dosya seçmeyin —": None,
                "NASA biçimi, 3 motor": "ornek_nasa_format.txt",
                "Başlıklı CSV (T24, T30… adlarıyla), 3 motor": "ornek_baslikli_3_motor.csv",
                "Sorunlu saha verisi (boş hücre, tekrar, karışık sıra, aralık dışı sensör)": "ornek_saha_verisi_sorunlu.csv",
                "Hatalı: eksik sensör sütunu": "ornek_hatali_eksik_sutun.csv",
                "Hatalı: metin içeren hücreler": "ornek_hatali_metin.csv"}
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("Sensör kaydını yükleyin", type=["txt", "csv"])
    ex = c2.selectbox("…veya bir örnek dosya deneyin", list(examples))

    file_bytes, fname = None, None
    if up is not None:
        file_bytes, fname = up.getvalue(), up.name
    elif examples[ex]:
        fname = examples[ex]
        file_bytes = (BASE / "ornek_veriler" / fname).read_bytes()

    if file_bytes is None:
        st.info("Başlamak için bir dosya yükleyin veya bir örnek dosya seçin.")
        st.stop()

    raw, msgs = read_uploaded(file_bytes, fname)
    clean, issues = (None, []) if raw is None else validate(raw, TRAIN_RANGES, WINDOW)
    st.markdown(f"##### Doğrulama raporu — `{fname}`")
    for level, text in msgs + issues:
        {"error": st.error, "warning": st.warning, "info": st.info}[level](text)
    if clean is None:
        st.stop()

    summary, per_cycle = summarize_engines(model, clean, CRIT, PLAN, WINDOW, CAP, TRAIN_RANGES)
    summary.insert(4, "durum", [f"{RISK_STYLE[r][0]} {r}" for r in summary["risk"]])

    st.markdown("##### Tahmin sonuçları")
    k = st.columns(3)
    for col, lvl in zip(k, ["Kritik", "Bakım planla", "Normal"]):
        col.metric(f"{RISK_STYLE[lvl][0]} {lvl}", int((summary["risk"] == lvl).sum()))
    st.dataframe(summary.drop(columns=["risk"]).sort_values("tahmini_RUL"), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Sonuçları indir (CSV)", to_csv_bytes(summary.drop(columns=["durum"])), "rul_tahminleri.csv", "text/csv")

    st.markdown("##### Motor detayı")
    u = st.selectbox("Motor", summary.sort_values("tahmini_RUL")["motor"].tolist())
    du = per_cycle[per_cycle["unit"] == u]
    row = summary[summary["motor"] == u].iloc[0]
    show_prediction_card(float(row["tahmini_RUL"]), int(row["kayıtlı_döngü"]))
    plot_rul_trajectory(du["cycle"].to_numpy(), du["RUL_pred"].to_numpy(), title=f"Motor {u}: tahminin zaman içindeki değişimi")
    sensors = st.multiselect("Sensörler", FEATURES_14, default=DEFAULT_SENSORS, format_func=sensor_label, key="up_sensors")
    plot_sensors(du, sensors)


# ======================================================================================
# 4) Ya şöyle olursa? (what-if)
# ======================================================================================
elif page == PAGES[3]:
    st.title("🧪 Ya şöyle olursa?")
    st.markdown("Bir test motorunun **son döngülerindeki** sensör değerlerini değiştirip modelin tahmininin nasıl değiştiğini görün. "
                "Hangi sensörün tahmine ne kadar etki ettiğini anlamak için basit bir duyarlılık analizidir.")
    c1, c2 = st.columns(2)
    unit = c1.selectbox("Motor", list(range(1, 101)), index=33, format_func=lambda u: f"Motor {u} (gerçek RUL {int(rul_true[u])})")
    s = c2.selectbox("Değiştirilecek sensör", FEATURES_14, index=FEATURES_14.index("s11"), format_func=sensor_label)
    c3, c4 = st.columns(2)
    shift = c3.slider("Değişim (eğitim verisindeki standart sapmanın katı)", -3.0, 3.0, 1.0, 0.25)
    last_k = c4.slider("Son kaç döngüye uygulansın?", 1, WINDOW, 20)

    base = test_df[test_df["unit"] == unit][INDEX_COLS + FEATURES_14].reset_index(drop=True)
    base[FEATURES_14] = base[FEATURES_14].astype(float)
    mod = base.copy()
    sd = TRAIN_RANGES[s]["std"]
    mod.loc[mod.index[-last_k:], s] += shift * sd

    with st.expander("✏️ Son 5 döngüyü elle düzenle (isteğe bağlı)"):
        edited = st.data_editor(mod.tail(5)[["cycle"] + FEATURES_14], disabled=["cycle"], hide_index=True, key="editor")
        mod.loc[mod.index[-5:], FEATURES_14] = edited[FEATURES_14].to_numpy()

    p0 = float(np.clip(model.predict(base), 0, CAP)[-1])
    p1 = float(np.clip(model.predict(mod), 0, CAP)[-1])
    l0, b0 = risk_badge(p0); l1, b1 = risk_badge(p1)
    k = st.columns(3)
    k[0].metric("Orijinal tahmin", f"{p0:.1f} döngü")
    k[1].metric("Değiştirilmiş tahmin", f"{p1:.1f} döngü", delta=f"{p1 - p0:+.1f}")
    k[2].markdown(f"**Risk**<br>{b0} → {b1}", unsafe_allow_html=True)
    st.caption(f"{sensor_label(s)} son {last_k} döngüde {shift:+.2f} × std ({shift * sd:+.3f}) değiştirildi.")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(base["cycle"], base[s], lw=1, label="orijinal"); ax.plot(mod["cycle"], mod[s], lw=1, alpha=.8, label="değiştirilmiş")
    ax.set_title(sensor_label(s)); ax.set_xlabel("döngü"); ax.legend(fontsize=8); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if st.checkbox("Tüm sensörler için duyarlılık tablosunu hesapla (+1 std, son 20 döngü)"):
        rows = []
        for sx in FEATURES_14:
            m2 = base.copy(); m2.loc[m2.index[-20:], sx] += TRAIN_RANGES[sx]["std"]
            rows.append({"sensör": sensor_label(sx), "tahmin değişimi (döngü)": float(np.clip(model.predict(m2), 0, CAP)[-1]) - p0})
        sens = pd.DataFrame(rows).sort_values("tahmin değişimi (döngü)")
        st.dataframe(sens.round(2), hide_index=True, use_container_width=True)
        st.caption("Negatif: sensörün artması kalan ömür tahminini düşürüyor (bozulma yönü).")


# ======================================================================================
# 5) Model performansı
# ======================================================================================
elif page == PAGES[4]:
    st.title("📊 Model performansı (resmi test seti, 100 motor)")
    preds = pd.read_csv(DATA / "results_stage12_test_predictions.csv")
    metrics = pd.read_csv(DATA / "results_stage12_test_metrics.csv", index_col=0)
    t1, t2, t3, t4 = st.tabs(["Metrikler", "Tahmin vs gerçek", "Özellik önemi", "Literatür"])

    with t1:
        st.dataframe(metrics.round(2), use_container_width=True)
        st.caption("RMSE/MAE/R²: gerçek RUL'a göre · NASA: 100 motor toplam PHM08 skoru (geç tahmini üstel cezalandırır) · "
                   "Geç tahmin %: ömrü 10 döngüden fazla fazla tahmin edilen motor oranı.")
        rob = pd.read_csv(DATA / "results_stage12_robustness.csv")
        st.markdown("**Sağlamlık testleri**")
        st.dataframe(rob.round(2), hide_index=True, use_container_width=True)

    with t2:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].scatter(preds["RUL_true"], preds["baseline_pred"], s=18, c="gray", alpha=.5, label="baseline (LR)")
        ax[0].scatter(preds["RUL_true"], preds["RUL_pred"], s=22, label=meta["final_model"])
        ax[0].plot([0, 150], [0, 150], "k--", lw=1); ax[0].set_xlabel("gerçek RUL"); ax[0].set_ylabel("tahmin"); ax[0].legend()
        ax[0].set_title("Tahmin vs gerçek")
        bins = pd.cut(preds["RUL_true"], [0, 25, 50, 75, 100, 125, 200], include_lowest=True)
        g = preds.groupby(bins, observed=True)["hata"].apply(lambda e: np.sqrt(np.mean(e ** 2)))
        ax[1].bar([str(i) for i in g.index], g.values); ax[1].set_title("Gerçek RUL aralığına göre RMSE"); ax[1].tick_params(axis="x", rotation=20)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown(f"**Risk sınıfı karışıklık matrisi** (kenar çubuğundaki eşiklerle: Kritik ≤ {CRIT}, Bakım planla ≤ {PLAN})")
        lv = ["Kritik", "Bakım planla", "Normal"]
        cm = pd.crosstab(pd.Categorical([risk_level(r, CRIT, PLAN) for r in preds["RUL_true"]], lv),
                         pd.Categorical([risk_level(r, CRIT, PLAN) for r in preds["RUL_pred"]], lv),
                         rownames=["gerçek"], colnames=["tahmin"], dropna=False)
        st.dataframe(cm, use_container_width=True)
        crit_n = int(cm.loc["Kritik"].sum())
        if crit_n:
            st.caption(f"Kritik motorları yakalama oranı: %{cm.loc['Kritik', 'Kritik'] / crit_n * 100:.0f} ({cm.loc['Kritik', 'Kritik']}/{crit_n}). "
                       "Eşikleri kenar çubuğundan değiştirerek etkisini görebilirsiniz.")

    with t3:
        imp = global_importance(model, FEAT_NAMES)
        if imp is None:
            st.info("Bu model türü yerleşik özellik önemi sunmuyor.")
        else:
            top = imp.head(15)[::-1]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top.index, top.values); ax.set_xlabel("önem payı (%)"); ax.set_title("En önemli 15 özellik (LightGBM gain)")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            kind = imp.groupby(lambda f: "ham" if "_" not in f else "".join(ch for ch in f.split("_")[1] if ch.isalpha())).sum()
            st.markdown("**Özellik türüne göre önem payı (%)**")
            st.bar_chart(kind.sort_values(ascending=False))
            st.caption("mean: kayan ortalama · std: kayan oynaklık · slope: kayan eğim (bozulma hızı) · ham: anlık sensör değeri")

    with t4:
        lit = pd.DataFrame([
            ("CNN — Babu vd. (2016)", 18.45), ("LSTM — Zheng vd. (2017)", 16.14), ("Tek katmanlı LSTM (2026)", 14.93),
            (f"{meta['final_model']} (bu proje)", cfg.get("test_RMSE", np.nan)),
            ("DCNN — Li vd. (2018)", 12.61), ("DA-CNN", 11.78), ("MS-DCNN", 11.44), ("DCGAN", 10.71)],
            columns=["yöntem", "RMSE"]).sort_values("RMSE", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(lit["yöntem"], lit["RMSE"], color=["tab:red" if "bu proje" in y else "tab:blue" for y in lit["yöntem"]])
        ax.set_xlabel("FD001 test RMSE (düşük daha iyi)"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Literatür değerleri yayımlanmış çalışmalardan alınmıştır; ön işleme ayrıntıları çalışmalar arasında farklılık gösterebilir.")
