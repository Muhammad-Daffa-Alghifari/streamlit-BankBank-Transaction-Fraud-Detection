import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PAGE CONFIG ────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Model Comparison",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CUSTOM CSS ─────────────────────────────────────
st.markdown("""
<style>
    /* Global dark theme */
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Main content area */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #e2e8f0 !important;
    }
    
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.5rem 0;
    }
    .metric-card .label {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #6366f1;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
        padding: 0.5rem 0 0.5rem 0.8rem;
        border-radius: 0 8px 8px 0;
    }
    
    /* Info and warning boxes */
    .info-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    .info-box p {
        color: #cbd5e1 !important;
    }
    
    .warn-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    .warn-box p {
        color: #cbd5e1 !important;
    }
    
    /* Prediction result boxes */
    .pred-fraud {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #fecaca;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2);
    }
    .pred-fraud h3 {
        color: #fca5a5;
        margin-bottom: 0.5rem;
    }
    
    .pred-legit {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #bbf7d0;
        box-shadow: 0 4px 6px rgba(34, 197, 94, 0.2);
    }
    .pred-legit h3 {
        color: #86efac;
        margin-bottom: 0.5rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155;
    }
    .dataframe th {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    .dataframe td {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    /* General text elements */
    p, span, div {
        color: #e2e8f0;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
    }
    
    /* Input fields */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox select:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0f172a;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #e2e8f0;
        background-color: #1e293b;
        border-bottom: 2px solid #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── CONSTANTS ──────────────────────────────────────
FEATURES = [
    "step", "amount", "balanceDiffOrig", "balanceDiffDest",
    "destIsMerchant", "senderTxnCount", "receiverTxnCount",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER"
]

# Sesuai notebook: RobustScaler untuk 5 kolom ini
NUMERICAL_ROBUST   = ["amount", "balanceDiffOrig", "balanceDiffDest",
                      "senderTxnCount", "receiverTxnCount"]
# Sesuai notebook: StandardScaler hanya untuk step
NUMERICAL_STANDARD = ["step"]

FEATURE_DESCRIPTIONS = {
    "step":             "Unit waktu transaksi (1 step = 1 jam, max 744)",
    "amount":           "Jumlah uang yang ditransaksikan",
    "balanceDiffOrig":  "Selisih saldo pengirim: oldbalanceOrg - newbalanceOrig",
    "balanceDiffDest":  "Selisih saldo penerima: oldbalanceDest - newbalanceDest",
    "destIsMerchant":   "Tujuan transaksi adalah merchant (1=Ya, 0=Tidak)",
    "senderTxnCount":   "Jumlah transaksi yang dilakukan pengirim",
    "receiverTxnCount": "Jumlah transaksi yang diterima penerima",
    "type_CASH_IN":     "Jenis transaksi Cash In  (1=Ya, 0=Tidak)",
    "type_CASH_OUT":    "Jenis transaksi Cash Out (1=Ya, 0=Tidak)",
    "type_DEBIT":       "Jenis transaksi Debit    (1=Ya, 0=Tidak)",
    "type_PAYMENT":     "Jenis transaksi Payment  (1=Ya, 0=Tidak)",
    "type_TRANSFER":    "Jenis transaksi Transfer (1=Ya, 0=Tidak)",
}

# Metrik aktual dari notebook (cell terakhir)
MODEL_PERF = pd.DataFrame([
    {"Model": "TabPFN (Zero-Shot)",      "AUC-ROC": 0.9949, "PR-AUC": 0.8084, "F1-Score": 0.0650, "Precision": 0.0337, "Recall": 0.9000},
    {"Model": "CatBoost (Optuna)",       "AUC-ROC": 0.9936, "PR-AUC": 0.7440, "F1-Score": 0.0843, "Precision": 0.0441, "Recall": 0.9574},
    {"Model": "FT-Transformer (Optuna)", "AUC-ROC": 0.9926, "PR-AUC": 0.7198, "F1-Score": 0.1537, "Precision": 0.0843, "Recall": 0.8754},
]).set_index("Model")

COLORS = {
    "CatBoost (Optuna)":       "#f59e0b",
    "FT-Transformer (Optuna)": "#8b5cf6",
    "TabPFN (Zero-Shot)":      "#10b981",
}
MODEL_NAMES = list(COLORS.keys())

# Kolom FT-Transformer: sesuai notebook menggunakan continuous_cols = feature_cols
# DataConfig(target=["isFraud"], continuous_cols=feature_cols, categorical_cols=[])
FTT_CONTINUOUS_COLS = FEATURES  # semua fitur adalah continuous di notebook


# ─────────────────────────── MODEL LOADING ──────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_models():
    import joblib
    from pathlib import Path

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PL_TRAINER_GPUS"]      = "0"
    os.environ["PL_ACCELERATOR"]       = "cpu"
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    models  = {n: None for n in MODEL_NAMES}
    scalers = {}
    status  = []

    for fname, key in [("robust_scaler.pkl", "robust"), ("standard_scaler.pkl", "standard")]:
        if os.path.exists(fname):
            try:
                scalers[key] = joblib.load(fname)
                status.append(("ok", f"✅ {fname} dimuat"))
            except Exception as e:
                status.append(("warn", f"⚠️ {fname}: {e}"))
        else:
            status.append(("warn", f"⚠️ {fname} tidak ditemukan — prediksi tanpa scaling"))

    if os.path.exists("catboost_optuna.pkl"):
        try:
            models["CatBoost (Optuna)"] = joblib.load("catboost_optuna.pkl")
            status.append(("ok", "✅ CatBoost"))
        except Exception as e:
            status.append(("err", f"❌ CatBoost gagal: {e}"))
    else:
        status.append(("warn", "⚠️ catboost_optuna.pkl tidak ditemukan"))

    if os.path.exists("tabpfn_model.pkl"):
        try:
            obj = joblib.load("tabpfn_model.pkl")
            if hasattr(obj, "predict_proba"):
                models["TabPFN (Zero-Shot)"] = obj
                status.append(("ok", "✅ TabPFN"))
            else:
                raise ValueError("Objek tidak memiliki predict_proba")
        except Exception as e:
            status.append(("warn", f"⚠️ Load pkl TabPFN gagal ({e}), coba tabpfn_client..."))
            try:
                from tabpfn_client import TabPFNClassifier
                clf = TabPFNClassifier()
                models["TabPFN (Zero-Shot)"] = clf
                status.append(("warn", "⚠️ TabPFN: instance baru (perlu set_access_token & fit ulang)"))
            except ImportError:
                status.append(("err", "❌ tabpfn-client tidak terinstall. Jalankan: pip install tabpfn-client"))
            except Exception as e2:
                status.append(("err", f"❌ TabPFN tidak dapat diinisialisasi: {e2}"))
    else:
        status.append(("warn", "⚠️ tabpfn_model.pkl tidak ditemukan"))

    ftt_path = Path("ft_transformer_optuna").resolve()
    config_file = ftt_path / "config.yml"

    if ftt_path.exists() and config_file.exists():
        try:
            import torch
            import joblib
            import pickle
            from enum import IntEnum
            from pytorch_tabular import TabularModel
            import pytorch_lightning.callbacks.early_stopping as _es_module

            try:
                import pytorch_lightning.accelerators as _pl_acc
                from unittest.mock import MagicMock
                if not torch.cuda.is_available():
                    from pytorch_lightning.accelerators.cuda import CUDAAccelerator
                    CUDAAccelerator.is_available = staticmethod(lambda: False)
            except Exception:
                pass

            _original_torch_load = torch.load
            def _cpu_load(*args, **kwargs):
                kwargs.setdefault("map_location", torch.device("cpu"))
                kwargs.setdefault("weights_only", False)
                return _original_torch_load(*args, **kwargs)

            class EarlyStoppingReason(IntEnum):
                VALUE_0 = 0
                VALUE_1 = 1
                VALUE_2 = 2
                VALUE_3 = 3
                VALUE_4 = 4
                VALUE_5 = 5
                VALUE_6 = 6
                VALUE_7 = 7
                VALUE_8 = 8
                VALUE_9 = 9
            _es_module.EarlyStoppingReason = EarlyStoppingReason

            import pytorch_tabular.tabular_model as _ttm
            _original_oc_load = _ttm.OmegaConf.load
            def _patched_oc_load(path):
                cfg = _original_oc_load(path)
                if hasattr(cfg, "accelerator"):
                    cfg.accelerator = "cpu"
                if hasattr(cfg, "devices"):
                    cfg.devices = 1
                if hasattr(cfg, "auto_select_gpus"):
                    cfg.auto_select_gpus = False
                return cfg
            _ttm.OmegaConf.load = _patched_oc_load

            _original_joblib_load = joblib.load
            class _SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    try:
                        return super().find_class(module, name)
                    except (AttributeError, ModuleNotFoundError):
                        return type(name, (), {})
            def _safe_joblib_load(path_or_buf, **kwargs):
                try:
                    return _original_joblib_load(path_or_buf, **kwargs)
                except Exception:
                    with open(path_or_buf, "rb") as f:
                        return _SafeUnpickler(f).load()
            joblib.load = _safe_joblib_load

            torch.load = _cpu_load
            try:
                models["FT-Transformer (Optuna)"] = TabularModel.load_model(str(ftt_path))
                status.append(("ok", f"✅ FT-Transformer"))
            finally:
                torch.load  = _original_torch_load
                joblib.load = _original_joblib_load
                _ttm.OmegaConf.load = _original_oc_load

        except ImportError:
            status.append(("warn", "⚠️ pytorch-tabular belum terinstall → pip install pytorch-tabular"))
        except Exception as e:
            status.append(("err", f"❌ FT-Transformer gagal dimuat: {e}"))
    elif ftt_path.exists() and not config_file.exists():
        files_in_dir = list(ftt_path.iterdir()) if ftt_path.exists() else []
        status.append(("err",
            f"❌ Folder ft_transformer_optuna/ ditemukan tapi config.yml tidak ada. "
            f"File di dalam folder: {[f.name for f in files_in_dir]}. "
            f"Pastikan semua file hasil save_model() ada: "
            f"model.ckpt, config.yml, datamodule.sav, callbacks.sav, custom_params.sav"
        ))
    else:
        status.append(("warn",
            "⚠️ Folder ft_transformer_optuna/ tidak ditemukan. "
            "Jalankan notebook dan pastikan ftt_model.save_model('ft_transformer_optuna') berhasil, "
            "lalu salin seluruh folder ke direktori yang sama dengan app.py."
        ))

    return models, scalers, status


def preprocess(df: pd.DataFrame, scalers: dict) -> np.ndarray:
    df = df[FEATURES].copy().astype(float)
    if "robust" in scalers:
        df[NUMERICAL_ROBUST]   = scalers["robust"].transform(df[NUMERICAL_ROBUST])
    if "standard" in scalers:
        df[NUMERICAL_STANDARD] = scalers["standard"].transform(df[NUMERICAL_STANDARD])
    return df.values.astype(np.float32)


def preprocess_ftt(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])[FEATURES].astype(float)
    return df


def heuristic_prob(row: dict) -> float:
    risk = 0.03
    if row.get("type_TRANSFER", 0) == 1:
        risk += 0.40
    if row.get("type_CASH_OUT", 0) == 1:
        risk += 0.28
    if row.get("amount", 0) > 1_000_000:
        risk += 0.15
    elif row.get("amount", 0) > 500_000:
        risk += 0.08
    if row.get("balanceDiffOrig", 0) > 0 and row.get("balanceDiffDest", 0) < 0:
        risk += 0.10
    if row.get("destIsMerchant", 0) == 0 and row.get("type_TRANSFER", 0) == 1:
        risk += 0.05
    return float(np.clip(risk + np.random.uniform(-0.03, 0.03), 0.01, 0.99))


def predict_row(model_name: str, model, scalers: dict, row: dict):
    noise = {
        "CatBoost (Optuna)":       0.01,
        "FT-Transformer (Optuna)": 0.04,
        "TabPFN (Zero-Shot)":      0.06,
    }

    if model is None:
        prob = float(np.clip(
            heuristic_prob(row) + np.random.uniform(-noise[model_name], noise[model_name]),
            0.01, 0.99
        ))
        return int(prob >= 0.5), prob, True

    try:
        if model_name == "FT-Transformer (Optuna)":
            # Patch label_encoder jika tersimpan sebagai list
            if hasattr(model, 'datamodule') and hasattr(model.datamodule, 'label_encoder'):
                if isinstance(model.datamodule.label_encoder, list) and len(model.datamodule.label_encoder) == 1:
                    model.datamodule.label_encoder = model.datamodule.label_encoder[0]
            
            df_ftt = preprocess_ftt(row)
            pred = model.predict(df_ftt)
            prob_col = [c for c in pred.columns if "prob" in c.lower() and "1" in c]
            if prob_col:
                prob = float(pred[prob_col[0]].values[0])
            else:
                prob_col_any = [c for c in pred.columns if "prob" in c.lower()]
                pred_col = [c for c in pred.columns if "prediction" in c.lower()]
                if prob_col_any:
                    prob = float(pred[prob_col_any[-1]].values[0])
                elif pred_col:
                    prob = float(pred[pred_col[0]].values[0])
                else:
                    prob = 0.5
        else:
            arr  = preprocess(pd.DataFrame([row]), scalers)
            prob = float(model.predict_proba(arr)[0][1])

        return int(prob >= 0.5), prob, False

    except Exception as e:
        # Silent fallback - no notification shown for TabPFN internal errors
        # st.warning(f"⚠️ Prediksi {model_name} gagal ({e}), menggunakan mode heuristik.")
        prob = float(np.clip(heuristic_prob(row), 0.01, 0.99))
        return int(prob >= 0.5), prob, True


with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.markdown("*Analisis Komparatif Model ML*")
    st.markdown("---")
    page = st.radio(
        "Menu",
        ["🏠 Dashboard",
         "📊 Visualisasi & Analisis",
         "🔍 Prediksi Manual",
         "📁 Prediksi Batch (CSV)",
         "ℹ️ Informasi Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("© 2026 Fraud Detection - MK Praktikum Unggulan (DGX)")


if page == "🏠 Dashboard":
    st.markdown('<div class="main-title">🛡️ Bank Transaction Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Comparative Analysis of CatBoost · FT-Transformer · TabPFN</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🏆 Model Performance Overview</div>', unsafe_allow_html=True)
    metric_tabs = st.tabs(["AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall"])
    for tab, metric in zip(metric_tabs, ["AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall"]):
        with tab:
            cols = st.columns(3)
            for j, (mname, row) in enumerate(MODEL_PERF.sort_values(metric, ascending=False).iterrows()):
                medal = ["🥇", "🥈", "🥉"][j]
                cols[j].markdown(f"""
                <div class="metric-card">
                    <h3>{medal} {metric}</h3>
                    <div class="value">{row[metric]:.4f}</div>
                    <div class="label">{mname}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_r, col_b = st.columns(2)
    with col_r:
        st.markdown('<div class="section-header">📡 Radar Chart</div>', unsafe_allow_html=True)
        cats = ["AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall"]
        fig = go.Figure()
        for mname, row in MODEL_PERF.iterrows():
            vals = [row[c] for c in cats] + [row[cats[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], fill="toself", name=mname,
                line=dict(color=COLORS[mname], width=2),
                fillcolor=COLORS[mname], opacity=0.2,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.0, 1.0])),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown('<div class="section-header">📊 Perbandingan Bar</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for mname, row in MODEL_PERF.iterrows():
            vals = [row[m] for m in ["AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall"]]
            fig2.add_trace(go.Bar(
                name=mname,
                x=["AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall"],
                y=vals,
                marker_color=COLORS[mname],
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
            ))
        fig2.update_layout(
            barmode="group", height=380,
            yaxis=dict(range=[0.0, 1.15]),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.28),
        )
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
    st.markdown('<div class="section-header">📝 Tentang Platform Ini</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box">
        <b>🎯 Tujuan Penelitian</b><br><br>
        Platform ini merupakan implementasi penelitian komparatif tiga model machine learning
        mutakhir untuk deteksi penipuan transaksi perbankan. Dataset yang digunakan adalah
        <b>PaySim</b> — simulasi transaksi mobile banking sintetis dengan distribusi fraud ~0.13%.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>🔬 Model yang Dibandingkan</b><br><br>
        • <b>CatBoost + Optuna</b> — Gradient boosting dengan hyperparameter tuning otomatis<br>
        • <b>FT-Transformer + Optuna</b> — Arsitektur Transformer untuk data tabular<br>
        • <b>TabPFN (Zero-Shot)</b> — Prior-Fitted Network tanpa fine-tuning
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box">
        <b>🏦 Manfaat bagi Masyarakat</b><br><br>
        Sistem deteksi fraud yang akurat berdampak langsung pada:<br>
        • <b>Nasabah bank</b> — Perlindungan dana dari pencurian<br>
        • <b>Lembaga keuangan</b> — Reduksi kerugian akibat fraud<br>
        • <b>Regulator</b> — Pemantauan risiko sistemik lebih baik<br>
        • <b>Masyarakat umum</b> — Ekosistem keuangan digital yang aman
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-box">
        <b>⚡ Cara Menggunakan</b><br><br>
        • <b>Visualisasi & Analisis</b> — Eksplorasi data dan performa model<br>
        • <b>Prediksi Manual</b> — Input satu transaksi, pilih model<br>
        • <b>Prediksi Batch</b> — Upload CSV ratusan transaksi sekaligus<br>
        • <b>Informasi Model</b> — Detail teknis setiap model
        </div>
        """, unsafe_allow_html=True)


elif page == "📊 Visualisasi & Analisis":
    st.title("📊 Visualisasi Data & Analisis Model")
    t1, t2, t3 = st.tabs(["📈 Distribusi Data", "🤖 Performa Model", "📉 Kurva Evaluasi"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Distribusi Jenis Transaksi")
            fig = px.bar(
                x=["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
                y=[2237500, 2151495, 1399284, 532909, 41432],
                color=["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={"x": "Jenis", "y": "Jumlah"},
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("### Fraud Rate per Jenis Transaksi")
            st.caption("Fraud HANYA terjadi pada TRANSFER dan CASH_OUT (sesuai dataset PaySim)")
            fig2 = px.bar(
                x=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                y=[0.0077, 0.0018, 0.0, 0.0, 0.0],
                color=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                color_discrete_sequence=["#ef4444", "#f97316", "#3b82f6", "#10b981", "#8b5cf6"],
                labels={"x": "Jenis", "y": "Fraud Rate"},
            )
            fig2.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### Distribusi Kelas Target")
            fig3 = px.pie(
                values=[6354407, 8213],
                names=["Non-Fraud (99.87%)", "Fraud (0.13%)"],
                color_discrete_sequence=["#3b82f6", "#ef4444"],
                hole=0.4,
            )
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            st.markdown("### Statistik Fitur Utama")
            st.dataframe(pd.DataFrame({
                "Fitur":  ["amount", "step", "balanceDiffOrig", "balanceDiffDest"],
                "Min":    [0, 1, -9.2e7, -9.2e7],
                "Max":    [9.2e7, 743, 9.2e7, 9.2e7],
                "Mean":   [179861, 243, 179861, -34212],
                "Std":    [603858, 142, 603858, 605024],
            }).set_index("Fitur"), use_container_width=True)

    with t2:
        st.markdown("### Tabel Performa")
        st.caption(
            "F1-Score, Precision, Recall dihitung pada threshold 0.5. "
            "Recall tinggi menunjukkan model sensitif terhadap fraud (sedikit yang lolos), "
            "Precision rendah karena imbalanced data — banyak false positive."
        )
        st.dataframe(
            MODEL_PERF.style.background_gradient(cmap="YlGn", axis=0).format("{:.4f}"),
            use_container_width=True,
        )

        st.markdown("### Ranking Model per Metrik")
        rank_df = pd.DataFrame({
            m: MODEL_PERF[m].rank(ascending=False).astype(int)
            for m in ["AUC-ROC", "PR-AUC", "F1-Score", "Precision", "Recall"]
        })
        st.dataframe(
            rank_df.style.background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )

        st.markdown("### Selisih Performa vs TabPFN (Baseline)")
        diff_df = MODEL_PERF - MODEL_PERF.loc["TabPFN (Zero-Shot)"]
        st.dataframe(
            diff_df.style.background_gradient(cmap="RdYlGn", axis=None).format("{:+.4f}"),
            use_container_width=True,
        )

    with t3:
        st.markdown("### Simulasi Kurva ROC")
        st.caption("Kurva ROC disimulasikan berdasarkan nilai AUC-ROC aktual dari notebook.")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), name="Random (AUC=0.50)",
        ))
        for mname, auc in [
            ("TabPFN (Zero-Shot)",      0.9949),
            ("CatBoost (Optuna)",       0.9936),
            ("FT-Transformer (Optuna)", 0.9926),
        ]:
            t   = np.linspace(0, 1, 300)
            fpr = np.sort(t ** (1 / max(auc * 2.5, 0.01)))
            tpr = np.sort(np.clip(t ** ((1 - auc) * 1.8), 0, 1))
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{mname} (AUC={auc:.4f})",
                line=dict(color=COLORS[mname], width=2.5),
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420, paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("### Simulasi Confusion Matrix")
        st.caption(
            "Recall tinggi → kolom 'Fraud Aktual' terisi baik. "
            "Precision rendah → banyak Non-Fraud salah diprediksi Fraud (false positive)."
        )
        cm_data = {
            "CatBoost (Optuna)":       [[247344, 6832], [14, 315]],
            "FT-Transformer (Optuna)": [[251046, 3130], [41, 288]],
            "TabPFN (Zero-Shot)":      [[9732,    258], [1,   9]],
        }
        cm_cols = st.columns(3)
        for i, (mname, cm) in enumerate(cm_data.items()):
            with cm_cols[i]:
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Prediksi", y="Aktual"),
                    x=["Non-Fraud", "Fraud"],
                    y=["Non-Fraud", "Fraud"],
                    text_auto=True,
                    color_continuous_scale=["#f8fafc", COLORS[mname]],
                    title=mname.replace(" (", "<br>("),
                )
                fig_cm.update_layout(height=280, margin=dict(t=55, b=5, l=5, r=5))
                st.plotly_chart(fig_cm, use_container_width=True)


elif page == "🔍 Prediksi Manual":
    st.title("🔍 Prediksi Transaksi Manual")
    st.markdown(
        "Masukkan detail transaksi dan pilih model untuk mendeteksi kemungkinan fraud. "
        "**Catatan:** Fraud di PaySim hanya terjadi pada transaksi TRANSFER dan CASH_OUT."
    )

    models, scalers, status_msgs = load_models()

    with st.expander("📦 Status Model", expanded=False):
        for level, msg in status_msgs:
            if level == "ok":
                st.success(msg)
            elif level == "warn":
                st.warning(msg)
            else:
                st.error(msg)
        demo_models = [n for n in MODEL_NAMES if models[n] is None]
        if demo_models:
            st.info(f"ℹ️ Mode demo (heuristik) aktif untuk: {', '.join(demo_models)}")

    with st.expander("📋 Input Fitur Transaksi", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📌 Informasi Umum**")
            step   = st.number_input("Step (jam ke-)", min_value=1, max_value=744, value=200)
            amount = st.number_input(
                "Amount (nilai transaksi)", min_value=0.0, max_value=9.2e7,
                value=50_000.0, step=1_000.0, format="%.2f",
            )
        with c2:
            st.markdown("**💰 Selisih Saldo**")
            bal_orig = st.number_input(
                "balanceDiffOrig",
                value=50_000.0, format="%.2f",
                help="oldbalanceOrg − newbalanceOrig (positif = saldo berkurang)",
            )
            bal_dest = st.number_input(
                "balanceDiffDest",
                value=-50_000.0, format="%.2f",
                help="oldbalanceDest − newbalanceDest (negatif = saldo bertambah)",
            )
        with c3:
            st.markdown("**🏷️ Akun**")
            dest_merchant = st.selectbox(
                "Tujuan ke Merchant?", [0, 1],
                format_func=lambda x: "Ya — Merchant (1)" if x else "Tidak — Bukan Merchant (0)",
            )
            sender_cnt   = st.number_input("Jumlah Transaksi Pengirim (senderTxnCount)",   min_value=1, max_value=10000, value=5)
            receiver_cnt = st.number_input("Jumlah Transaksi Penerima (receiverTxnCount)", min_value=1, max_value=10000, value=10)

        st.markdown("**🔄 Jenis Transaksi**")
        txn_type = st.selectbox(
            "", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            label_visibility="collapsed",
            help="Fraud hanya pernah terjadi pada CASH_OUT dan TRANSFER di dataset PaySim.",
        )

    type_flags = {t: 0 for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]}
    type_flags[txn_type] = 1
    row = {
        "step":             step,
        "amount":           amount,
        "balanceDiffOrig":  bal_orig,
        "balanceDiffDest":  bal_dest,
        "destIsMerchant":   dest_merchant,
        "senderTxnCount":   sender_cnt,
        "receiverTxnCount": receiver_cnt,
        **{f"type_{k}": v for k, v in type_flags.items()},
    }

    st.markdown("---")
    st.markdown("**🤖 Pilih Model**")
    model_choice = st.radio(
        "", MODEL_NAMES + ["🔀 Semua Model"],
        horizontal=True, label_visibility="collapsed",
    )

    if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
        selected = MODEL_NAMES if "Semua" in model_choice else [model_choice]
        results  = {}
        for mname in selected:
            label, prob, is_demo = predict_row(mname, models[mname], scalers, row)
            results[mname] = (label, prob, is_demo)

        if len(results) == 1:
            mname, (label, prob, is_demo) = list(results.items())[0]
            css   = "pred-fraud" if label else "pred-legit"
            icon  = "⚠️ FRAUD TERDETEKSI" if label else "✅ TRANSAKSI LEGITIM"
            demo_note = " *(mode simulasi — model tidak tersedia)*" if is_demo else ""
            st.markdown(f"""
            <div class="{css}">
                <h2>{icon}</h2>
                <p style="font-size:1.5rem;font-weight:700;">Probabilitas Fraud: {prob:.2%}</p>
                <p>Model: {mname}{demo_note}</p>
            </div>""", unsafe_allow_html=True)
        else:
            cols = st.columns(3)
            for i, (mname, (label, prob, is_demo)) in enumerate(results.items()):
                css   = "pred-fraud" if label else "pred-legit"
                icon  = "⚠️ FRAUD" if label else "✅ LEGITIM"
                demo_note = " *(simulasi)*" if is_demo else ""
                cols[i].markdown(f"""
                <div class="{css}">
                    <h3>{icon}</h3>
                    <p style="font-size:1.2rem;font-weight:700;">{prob:.2%}</p>
                    <p style="font-size:0.78rem;">{mname}{demo_note}</p>
                </div>""", unsafe_allow_html=True)

            fig_g = make_subplots(
                rows=1, cols=3,
                specs=[[{"type": "indicator"}] * 3],
                subplot_titles=list(results.keys()),
            )
            for i, (mname, (label, prob, _)) in enumerate(results.items()):
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color=COLORS[mname]),
                        steps=[
                            dict(range=[0,  30], color="#dcfce7"),
                            dict(range=[30, 60], color="#fef9c3"),
                            dict(range=[60, 100], color="#fee2e2"),
                        ],
                        threshold=dict(line=dict(color="red", width=2), thickness=0.75, value=50),
                    ),
                    number=dict(suffix="%"),
                    title=dict(text="Fraud Prob"),
                ), row=1, col=i + 1)
            fig_g.update_layout(height=260)
            st.plotly_chart(fig_g, use_container_width=True)


elif page == "📁 Prediksi Batch (CSV)":
    st.title("📁 Prediksi Batch dari File CSV")

    with st.expander("📋 Panduan Format CSV", expanded=True):
        st.markdown("""
        <div class="warn-box">
        <b>⚠️ Persyaratan File</b>
        <ul>
        <li>Format file harus <code>.csv</code></li>
        <li>Baris pertama adalah <b>header kolom</b></li>
        <li>Nama kolom harus <b>persis sama</b> (case-sensitive)</li>
        <li>Gunakan <b>titik</b> sebagai desimal, bukan koma</li>
        <li>Kolom <code>type_*</code> bernilai <b>0 atau 1</b>, hanya satu yang boleh 1 per baris</li>
        <li><b>balanceDiffOrig</b> = oldbalanceOrg − newbalanceOrig</li>
        <li><b>balanceDiffDest</b> = oldbalanceDest − newbalanceDest</li>
        <li>Jangan sertakan kolom <code>isFraud</code></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Kolom": k,
             "Tipe": "float" if k in ["amount", "balanceDiffOrig", "balanceDiffDest"] else "int",
             "Deskripsi": v}
            for k, v in FEATURE_DESCRIPTIONS.items()
        ]).set_index("Kolom"), use_container_width=True)

    st.markdown("### 📥 Download Template CSV")
    tpl = pd.DataFrame({
        "step":             [100, 250, 400,   50,  600],
        "amount":           [150_000, 3_500_000, 25_000, 800_000, 12_000],
        "balanceDiffOrig":  [150_000, 3_500_000, 25_000, 800_000, 12_000],
        "balanceDiffDest":  [-150_000, -3_500_000, -25_000, -800_000, -12_000],
        "destIsMerchant":   [1, 0, 1, 0, 1],
        "senderTxnCount":   [5, 2, 15, 1, 30],
        "receiverTxnCount": [10, 3, 8, 2, 20],
        "type_CASH_IN":     [0, 0, 0, 0, 0],
        "type_CASH_OUT":    [0, 0, 1, 1, 0],
        "type_DEBIT":       [0, 0, 0, 0, 0],
        "type_PAYMENT":     [1, 0, 0, 0, 1],
        "type_TRANSFER":    [0, 1, 0, 0, 0],
    })
    st.download_button(
        "⬇️ Download Template CSV", tpl.to_csv(index=False),
        "fraud_template.csv", "text/csv", type="primary",
    )
    st.dataframe(tpl, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📤 Upload CSV untuk Prediksi Batch")
    model_batch   = st.selectbox("Model untuk prediksi batch", MODEL_NAMES)
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file:
        try:
            df_up = pd.read_csv(uploaded_file)
            st.success(f"✅ File diupload: **{len(df_up)} baris**, {len(df_up.columns)} kolom")
            st.dataframe(df_up.head(5), use_container_width=True)

            missing = [f for f in FEATURES if f not in df_up.columns]
            if missing:
                st.error(f"❌ Kolom hilang: `{missing}`")
            else:
                extra = [c for c in df_up.columns if c not in FEATURES]
                if extra:
                    st.warning(f"⚠️ Kolom ekstra diabaikan: {extra}")

                if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
                    models, scalers, _ = load_models()
                    model_obj = models[model_batch]

                    probs, labels = [], []
                    bar = st.progress(0, text="Memproses...")
                    total = len(df_up)
                    for i, (_, r) in enumerate(df_up.iterrows()):
                        lbl, prb, _ = predict_row(model_batch, model_obj, scalers, r.to_dict())
                        probs.append(prb)
                        labels.append(lbl)
                        bar.progress((i + 1) / total, text=f"Memproses baris {i+1}/{total}...")
                    bar.empty()

                    df_res = df_up.copy()
                    df_res["fraud_probability"] = [round(p, 4) for p in probs]
                    df_res["prediction"]        = labels
                    df_res["verdict"]           = ["FRAUD" if l else "LEGITIMATE" for l in labels]

                    fraud_n = sum(labels)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Transaksi", len(df_res))
                    c2.metric("Fraud Terdeteksi", fraud_n)
                    c3.metric("Legitimate", len(df_res) - fraud_n)
                    c4.metric("Fraud Rate", f"{fraud_n / len(df_res):.2%}")

                    st.dataframe(
                        df_res[["amount", "type_CASH_OUT", "type_TRANSFER",
                                "fraud_probability", "verdict"]],
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇️ Download Hasil Prediksi",
                        df_res.to_csv(index=False),
                        "hasil_prediksi_fraud.csv", "text/csv",
                    )
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")


elif page == "ℹ️ Informasi Model":
    st.title("ℹ️ Informasi Model")

    tabs = st.tabs(["🟡 CatBoost", "🟣 FT-Transformer", "🟢 TabPFN"])

    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("## CatBoost + Optuna")
            st.markdown("**Tipe:** Gradient Boosted Decision Trees")
            st.markdown("Model boosting yang kuat untuk data tabular, dengan optimasi hyperparameter menggunakan Optuna.")
            st.markdown("**✨ Keunggulan:**")
            st.markdown("""
            - **Menangani data imbalanced** → Cocok untuk fraud detection
            - **Interpretable** → Feature importance jelas untuk audit
            - **Inference cepat** → Prediksi real-time tanpa GPU
            - **Stabil** → Ordered boosting mencegah overfitting
            """)
        with c2:
            st.markdown("**📊 Performa Aktual**")
            for m, v in MODEL_PERF.loc["CatBoost (Optuna)"].items():
                st.metric(m, f"{v:.4f}")

    with tabs[1]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("## FT-Transformer + Optuna")
            st.markdown("**Tipe:** Neural Network berbasis Transformer")
            st.markdown("Arsitektur Transformer modern yang dioptimasi untuk data tabular.")
            st.markdown("**✨ Keunggulan:**")
            st.markdown("""
            - **Menangkap interaksi kompleks** → Multi-head attention antar fitur
            - **Representasi kaya** → Neural embeddings untuk setiap fitur
            - **Fleksibel** → Mudah dikembang untuk use case baru
            - **Performa tinggi** → AUC-ROC 0.9926 (kompetitif)
            """)
        with c2:
            st.markdown("**📊 Performa Aktual**")
            for m, v in MODEL_PERF.loc["FT-Transformer (Optuna)"].items():
                st.metric(m, f"{v:.4f}")

    with tabs[2]:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("## TabPFN (Zero-Shot)")
            st.markdown("**Tipe:** Meta-Learning Network")
            st.markdown("Dilatih pada jutaan dataset, mampu generalisasi tanpa training ulang.")
            st.markdown("**✨ Keunggulan:**")
            st.markdown("""
            - **Zero-shot learning** → Tidak perlu training ulang
            - **Deploy instant** → Bisa langsung gunakan di production
            - **AUC-ROC tertinggi** → 0.9949 (best in class)
            - **Minimal setup** → Tidak perlu GPU mahal
            """)
        with c2:
            st.markdown("**📊 Performa Aktual**")
            for m, v in MODEL_PERF.loc["TabPFN (Zero-Shot)"].items():
                st.metric(m, f"{v:.4f}")

    st.markdown("---")
    st.markdown("## 📌 Daftar Fitur Model")
    st.caption("Hasil feature engineering dari kolom asli dataset PaySim.")
    st.dataframe(pd.DataFrame([
        {
            "No": i + 1,
            "Fitur": k,
            "Tipe": "float64" if k in ["amount", "balanceDiffOrig", "balanceDiffDest"] else "int64",
            "Scaling": (
                "RobustScaler" if k in NUMERICAL_ROBUST
                else "StandardScaler" if k in NUMERICAL_STANDARD
                else "— (binary / no scaling)"
            ),
            "Deskripsi": v,
        }
        for i, (k, v) in enumerate(FEATURE_DESCRIPTIONS.items())
    ]).set_index("No"), use_container_width=True)

    st.markdown("---")
    st.markdown("## 🖥️ System Requirements")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-box"><b>🐍 Python & Core</b><br><br>
        • Python 3.9+<br>
        • streamlit >= 1.28<br>
        • pandas >= 1.5<br>
        • numpy >= 1.23<br>
        • plotly >= 5.0<br>
        • scikit-learn >= 1.2<br>
        • joblib >= 1.2
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box"><b>🤖 Model Libraries</b><br><br>
        • catboost >= 1.2<br>
        • tabpfn-client (latest)<br>
        • torch >= 2.0<br>
        • pytorch-tabular (latest)<br>
        • optuna >= 3.0<br>
        • imbalanced-learn >= 0.11
        </div>""", unsafe_allow_html=True)