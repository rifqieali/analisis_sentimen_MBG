"""
Aplikasi Pengujian Real-Time ABSA MBG
Memuat model terlatih dari joblib — tidak perlu training ulang.
Jalankan: streamlit run app_realtime.py
"""

import os
import re
import string
import warnings
import requests
import joblib
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen MBG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0D1117;
    --bg-secondary: #161B22;
    --bg-card: #1C2128;
    --bg-card-hover: #212830;
    --border: #30363D;
    --border-accent: #388BFD;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --text-muted: #6E7681;
    --green: #3FB950;
    --green-dim: #1B3B22;
    --red: #F85149;
    --red-dim: #3B1B1B;
    --blue: #388BFD;
    --blue-dim: #1B2B3B;
    --yellow: #D29922;
    --yellow-dim: #3B2B0D;
    --radius: 10px;
    --radius-sm: 6px;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp { background-color: var(--bg-primary); }

/* Header utama */
.main-header {
    background: linear-gradient(135deg, #1C2128 0%, #161B22 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #388BFD, #3FB950, #D29922);
}
.main-header h1 {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin: 0;
}

/* Card */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 12px;
}

/* Badge sentimen */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-positif { background: var(--green-dim); color: var(--green); border: 1px solid #2D5A3D; }
.badge-negatif { background: var(--red-dim); color: var(--red); border: 1px solid #5A2D2D; }
.badge-lainnya { background: var(--blue-dim); color: var(--blue); border: 1px solid #2D4A6A; }

/* Metric card */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
    line-height: 1.2;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}
.metric-model {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Input area */
.stTextArea textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--border-accent) !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.1) !important;
}

/* Tombol */
.stButton > button {
    background: var(--blue) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #4D97FF !important;
    transform: translateY(-1px) !important;
}

/* Radio */
.stRadio > div { gap: 8px !important; }
.stRadio label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}

/* Tabel */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* Divider */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* Info / warning / success */
.stAlert { border-radius: var(--radius) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* File uploader */
.stFileUploader {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Download button */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    border-color: var(--border-accent) !important;
    background: var(--bg-card-hover) !important;
}

/* Result table styling */
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.result-table th {
    background: var(--bg-secondary);
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.result-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
    vertical-align: top;
}
.result-table tr:last-child td { border-bottom: none; }
.result-table tr:hover td { background: var(--bg-card-hover); }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
.correct { color: var(--green); }
.wrong { color: var(--red); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# KONSTANTA
# ============================================================
ASPEK_DICT = {
    'Kualitas': [
        'kualitas', 'bagus', 'jelek', 'enak', 'basi', 'gizi', 'susu',
        'menu', 'rasa', 'porsi', 'higienis', 'keracunan', 'sehat',
        'mentah', 'keras', 'hambar', 'ulat', 'lauk', 'sayur',
        'karbohidrat', 'protein', 'lemak', 'gula', 'ayam', 'telur',
        'kenyang', 'alergi', 'higienitas'
    ],
    'Layanan': [
        'layan', 'antri', 'ramah', 'lambat', 'cepat', 'bantu', 'saji',
        'distribusi', 'vendor', 'katering', 'sekolah', 'siswa', 'guru',
        'telat', 'molor', 'bocor', 'tepat waktu', 'pelosok', 'merata',
        'zonasi', 'umkm', 'kemasan', 'kotak', 'plastik'
    ],
    'Anggaran': [
        'harga', 'mahal', 'murah', 'biaya', 'bayar', 'anggar', 'boros',
        'korupsi', 'dana', 'apbn', 'pajak', 'potong', 'sunat', 'markup',
        'tender', 'proyek', 'apbd', 'defisit', 'utang', 'ekonomi',
        'alokasi', 'transparan'
    ],
}
KONJUNGSI     = r'\b(tetapi|namun|meskipun|tapi|sedangkan|cuman|cuma|sayangnya|padahal|walau|walaupun|pasalnya)\b'
KONJUNGSI_SET = {'tetapi','namun','meskipun','tapi','sedangkan','cuman','cuma','sayangnya','padahal','walau','walaupun','pasalnya'}
JOBLIB_PATH   = "saved_model_data.joblib"


# ============================================================
# RESOURCE LOADING
# ============================================================
@st.cache_resource
def load_nlp_resources():
    factory_stem = StemmerFactory()
    stemmer      = factory_stem.create_stemmer()
    factory_stop = StopWordRemoverFactory()
    sw_sastrawi  = set(factory_stop.get_stop_words())
    sw_custom    = {
        "yg","dg","rt","dgn","ny","d","klo","kalo","amp",
        "biar","bikin","udah","udh","aja","sih","deh","nih",
        "lah","dong","kan","tuh","mah","wkwk","haha","hehe",
        "aku","saya","kamu","dia","kita","kami","mereka","sama"
    }
    negation_words = {'tidak','tak','tiada','bukan','jangan','belum','kurang','gak','ga','nggak','enggak'}
    final_stopwords = (sw_sastrawi | sw_custom) - negation_words
    return stemmer, final_stopwords, negation_words

@st.cache_data
def load_norm_dict():
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    try:
        resp = requests.get(url, timeout=15)
        xls  = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
        cols = xls.columns.tolist()
        return dict(zip(xls[cols[0]].astype(str).str.lower(), xls[cols[1]].astype(str).str.lower()))
    except Exception:
        return {"yg":"yang","gk":"tidak","ga":"tidak","gak":"tidak","bgt":"banget"}

@st.cache_resource
def load_model():
    if not os.path.exists(JOBLIB_PATH):
        return None
    return joblib.load(JOBLIB_PATH)

stemmer, final_stopwords, negation_words = load_nlp_resources()
norm_dict = load_norm_dict()
model_data = load_model()


# ============================================================
# FUNGSI PREPROCESSING
# ============================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def normalize_text(text: str) -> str:
    return ' '.join(norm_dict.get(w, w) for w in text.split())

def segmentasi_kalimat(text: str) -> list:
    parts = re.split(KONJUNGSI, text)
    return [s.strip() for s in parts if s.strip() and s.strip() not in KONJUNGSI_SET] or [text]

def stopword_and_stem(text: str) -> str:
    words = [w for w in text.split() if w not in final_stopwords]
    return stemmer.stem(' '.join(words))

def preprocess_text(text: str) -> list:
    cleaned  = clean_text(text)
    normed   = normalize_text(cleaned)
    segments = segmentasi_kalimat(normed)
    return [s for seg in segments if (s := stopword_and_stem(seg)).strip()]

def get_aspects(text: str) -> list:
    tokens = set(str(text).split())
    found  = [asp for asp, keys in ASPEK_DICT.items() if not tokens.isdisjoint(keys)]
    return found if found else ['Lainnya']


# ============================================================
# FUNGSI ANALISIS
# ============================================================
def analyze_texts(texts: list, nb_model, svm_model, vec) -> pd.DataFrame:
    rows = []
    for text in texts:
        segments = preprocess_text(text)
        if not segments:
            continue
        for seg in segments:
            X         = vec.transform([seg])
            pred_nb   = nb_model.predict(X)[0]
            pred_svm  = svm_model.predict(X)[0]
            probs_nb  = nb_model.predict_proba(X)[0]
            classes   = nb_model.classes_
            prob_str  = " | ".join([f"{c}: {probs_nb[j]:.1%}" for j, c in enumerate(classes)])
            rows.append({
                "Teks Asli"    : text,
                "Segmen Bersih": seg,
                "Aspek"        : ", ".join(get_aspects(seg)),
                "Prediksi SVM" : pred_svm,
                "Prediksi NB"  : pred_nb,
                "Probabilitas NB": prob_str,
            })
    return pd.DataFrame(rows)


# ============================================================
# FUNGSI VISUALISASI
# ============================================================
def _plt_style():
    plt.rcParams.update({
        'figure.facecolor': '#1C2128',
        'axes.facecolor'  : '#1C2128',
        'axes.edgecolor'  : '#30363D',
        'axes.labelcolor' : '#8B949E',
        'xtick.color'     : '#8B949E',
        'ytick.color'     : '#8B949E',
        'text.color'      : '#E6EDF3',
        'grid.color'      : '#30363D',
        'grid.alpha'      : 0.4,
    })

def plot_distribusi_pie(df_res: pd.DataFrame):
    _plt_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#1C2128')
    for ax, col, title in [
        (axes[0], 'Prediksi SVM', 'LinearSVC'),
        (axes[1], 'Prediksi NB',  'Multinomial NB'),
    ]:
        counts = df_res[col].value_counts()
        colors = ['#3FB950' if l == 'Positif' else '#F85149' for l in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops=dict(edgecolor='#1C2128', linewidth=2),
            textprops=dict(color='#E6EDF3', fontsize=10)
        )
        for at in autotexts:
            at.set_fontweight('bold')
            at.set_fontsize(11)
        ax.set_title(title, color='#E6EDF3', fontweight='bold', fontsize=12, pad=12)
    plt.tight_layout()
    return fig

def plot_aspek_bar(df_res: pd.DataFrame):
    _plt_style()
    df_asp = df_res[~df_res['Aspek'].str.contains('Lainnya', na=False)].copy()
    if df_asp.empty:
        return None
    pivot = df_asp.groupby(['Aspek', 'Prediksi SVM']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1C2128')
    colors = {'Positif': '#3FB950', 'Negatif': '#F85149'}
    pivot.plot(
        kind='bar', ax=ax,
        color=[colors.get(c, '#388BFD') for c in pivot.columns],
        edgecolor='#1C2128', linewidth=1.5, width=0.6
    )
    ax.set_title('Distribusi Sentimen per Aspek (LinearSVC)',
                 color='#E6EDF3', fontweight='bold', pad=12)
    ax.set_xlabel(''); ax.set_ylabel('Jumlah Segmen', color='#8B949E')
    ax.legend(title='Sentimen', labelcolor='#E6EDF3',
              facecolor='#161B22', edgecolor='#30363D',
              title_fontsize=9, fontsize=9)
    plt.xticks(rotation=0)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width()/2, p.get_height() + 0.3),
                        ha='center', va='bottom', fontsize=9, color='#E6EDF3')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_wordcloud(df_res: pd.DataFrame):
    _plt_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1C2128')
    for ax, label, cmap in [
        (axes[0], 'Positif', 'Greens'),
        (axes[1], 'Negatif', 'Reds'),
    ]:
        teks = " ".join(df_res[df_res['Prediksi SVM'] == label]['Segmen Bersih'].astype(str))
        if teks.strip():
            wc = WordCloud(
                width=500, height=280,
                background_color='#1C2128',
                colormap=cmap,
                max_words=60,
                prefer_horizontal=0.9
            ).generate(teks)
            ax.imshow(wc, interpolation='bilinear')
        else:
            ax.text(0.5, 0.5, f'Tidak ada data\n{label}',
                    ha='center', va='center', color='#6E7681',
                    transform=ax.transAxes, fontsize=11)
        ax.axis('off')
        ax.set_title(f'WordCloud — {label} (SVM)',
                     color='#E6EDF3', fontweight='bold', pad=8)
    plt.tight_layout()
    return fig

def plot_akurasi_bar(acc_svm: float, acc_nb: float):
    _plt_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1C2128')
    models = ['LinearSVC', 'Multinomial NB']
    accs   = [acc_svm, acc_nb]
    colors = ['#388BFD', '#3FB950']
    bars   = ax.bar(models, accs, color=colors, width=0.45,
                    edgecolor='#1C2128', linewidth=1.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Akurasi', color='#8B949E')
    ax.set_title('Akurasi Pengujian Real-Time', color='#E6EDF3',
                 fontweight='bold', pad=12)
    ax.axhline(y=0.7, color='#D29922', linestyle='--',
               linewidth=1.2, label='Threshold 70%', alpha=0.8)
    ax.legend(labelcolor='#E6EDF3', facecolor='#161B22',
              edgecolor='#30363D', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', fontsize=13,
                fontweight='bold', color='#E6EDF3')
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred_svm, y_pred_nb, labels_cm):
    _plt_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#1C2128')
    for ax, y_pred, title, cmap in [
        (axes[0], y_pred_svm, 'LinearSVC',     'Blues'),
        (axes[1], y_pred_nb,  'Multinomial NB', 'Greens'),
    ]:
        cm = confusion_matrix(y_true, y_pred, labels=labels_cm)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=labels_cm, yticklabels=labels_cm,
                    ax=ax, linewidths=0.5, linecolor='#30363D',
                    annot_kws={'fontsize': 13, 'fontweight': 'bold'})
        ax.set_title(f'Confusion Matrix — {title}',
                     color='#E6EDF3', fontweight='bold', pad=10)
        ax.set_xlabel('Prediksi', color='#8B949E')
        ax.set_ylabel('Aktual', color='#8B949E')
    plt.tight_layout()
    return fig


# ============================================================
# HEADER UTAMA
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🍽️ Analisis Sentimen MBG</h1>
    <p>Pengujian real-time menggunakan model Multinomial Naïve Bayes & LinearSVC yang telah dilatih</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CEK MODEL
# ============================================================
if model_data is None:
    st.error(f"File `{JOBLIB_PATH}` tidak ditemukan. Letakkan file joblib di folder yang sama dengan aplikasi ini.")
    st.info("Jalankan pipeline Colab terlebih dahulu untuk menghasilkan file joblib, lalu download dan letakkan di sini.")
    st.stop()

nb_model  = model_data['model_nb']
svm_model = model_data['model_svm']
vec       = model_data['vectorizer']

# Info model
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown("""<div class="card">
        <div class="card-title">Model Terlatih</div>
        <div style="color:#E6EDF3;font-weight:700;">MultinomialNB + LinearSVC</div>
        <div style="color:#8B949E;font-size:0.82rem;margin-top:4px;">Dimuat dari saved_model_data.joblib</div>
    </div>""", unsafe_allow_html=True)
with col_info2:
    vocab_size = len(vec.vocabulary_)
    st.markdown(f"""<div class="card">
        <div class="card-title">TF-IDF Vocabulary</div>
        <div style="color:#E6EDF3;font-weight:700;font-family:'JetBrains Mono',monospace;">{vocab_size:,} fitur</div>
        <div style="color:#8B949E;font-size:0.82rem;margin-top:4px;">ngram (1,2) · max_features 3000</div>
    </div>""", unsafe_allow_html=True)
with col_info3:
    classes = nb_model.classes_
    st.markdown(f"""<div class="card">
        <div class="card-title">Kelas Prediksi</div>
        <div style="color:#E6EDF3;font-weight:700;">{' · '.join(classes)}</div>
        <div style="color:#8B949E;font-size:0.82rem;margin-top:4px;">3 aspek teridentifikasi</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ============================================================
# SESSION STATE
# ============================================================
for key, default in [
    ('rt_analyzed', False),
    ('rt_texts', []),
    ('rt_labels', []),
    ('rt_has_labels', False),
    ('df_results', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# INPUT PANEL
# ============================================================
if not st.session_state['rt_analyzed']:
    mode = st.radio(
        "Mode Input:",
        ("✏️ Input Manual (Paste Teks)", "📄 Upload CSV"),
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if "Manual" in mode:
        st.markdown("""<div class="card">
            <div class="card-title">Panduan Input</div>
            <div style="color:#8B949E;font-size:0.85rem;">
                Satu kalimat per baris. Untuk evaluasi akurasi, tambahkan label dengan format:<br>
                <code style="color:#388BFD;">kalimat | Positif</code> atau <code style="color:#388BFD;">kalimat | Negatif</code><br>
                Tanpa label pun bisa — hanya prediksi yang ditampilkan.
            </div>
        </div>""", unsafe_allow_html=True)

        raw = st.text_area(
            "Masukkan kalimat uji:",
            height=280,
            placeholder="makanan bergizi enak dan porsinya cukup untuk anak sekolah | Positif\ndistribusi makanan sering telat siswa menunggu berjam-jam | Negatif\ndana MBG dikorupsi dan dimarkup oknum tidak bertanggung jawab | Negatif\nmenu MBG lezat dan higienis sangat membantu gizi siswa",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            run = st.button("🔍 Analisis Teks", use_container_width=True)

        if run:
            if raw.strip():
                texts, labels, has_labels = [], [], False
                for line in raw.split('\n'):
                    line = line.strip()
                    if not line: continue
                    if '|' in line:
                        parts = line.split('|', 1)
                        texts.append(parts[0].strip())
                        labels.append(parts[1].strip().capitalize())
                        has_labels = True
                    else:
                        texts.append(line)
                        labels.append(None)
                st.session_state.update({
                    'rt_texts': texts, 'rt_labels': labels,
                    'rt_has_labels': has_labels, 'rt_analyzed': True
                })
                st.rerun()
            else:
                st.warning("Teks tidak boleh kosong.")

    else:  # Upload CSV
        st.markdown("""<div class="card">
            <div class="card-title">Format CSV</div>
            <div style="color:#8B949E;font-size:0.85rem;">
                Kolom wajib: <code style="color:#388BFD;">teks</code> (kalimat uji) dan opsional <code style="color:#388BFD;">label</code> (Positif/Negatif) untuk evaluasi akurasi.
            </div>
        </div>""", unsafe_allow_html=True)

        template_df = pd.DataFrame({
            "teks" : [
                "makanan bergizi enak dan porsinya cukup untuk anak sekolah",
                "distribusi makanan sering telat siswa menunggu berjam-jam",
                "dana MBG dikorupsi dan dimarkup oknum tidak bertanggung jawab",
                "kualitas makanan bagus dan anak kenyang setelah makan",
                "vendor katering tidak profesional dan pengiriman selalu molor",
            ],
            "label": ["Positif", "Negatif", "Negatif", "Positif", "Negatif"]
        })

        col_dl, col_up = st.columns([1, 2])
        with col_dl:
            st.download_button(
                "📥 Download Template CSV",
                data=template_df.to_csv(index=False).encode('utf-8'),
                file_name="template_pengujian.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_up:
            uploaded_test = st.file_uploader(
                "Upload CSV:", type="csv", label_visibility="collapsed"
            )

        col_btn3, col_btn4 = st.columns([1, 4])
        with col_btn3:
            run_csv = st.button("🔍 Analisis CSV", use_container_width=True)

        if run_csv:
            if uploaded_test:
                df_csv = pd.read_csv(uploaded_test)
                if 'teks' not in df_csv.columns:
                    st.error("Kolom 'teks' tidak ditemukan.")
                else:
                    texts  = df_csv['teks'].dropna().astype(str).tolist()
                    labels = df_csv['label'].tolist() if 'label' in df_csv.columns else [None]*len(texts)
                    has_labels = 'label' in df_csv.columns
                    st.session_state.update({
                        'rt_texts': texts, 'rt_labels': labels,
                        'rt_has_labels': has_labels, 'rt_analyzed': True
                    })
                    st.rerun()
            else:
                st.warning("Upload CSV terlebih dahulu.")


# ============================================================
# HASIL ANALISIS
# ============================================================
else:
    # Tombol ulangi
    col_ulang, _ = st.columns([1, 5])
    with col_ulang:
        if st.button("↩ Ulangi Analisis", use_container_width=True):
            st.session_state.update({
                'rt_analyzed': False, 'rt_texts': [],
                'rt_labels': [], 'rt_has_labels': False, 'df_results': None
            })
            st.rerun()

    texts      = st.session_state['rt_texts']
    labels     = st.session_state['rt_labels']
    has_labels = st.session_state['rt_has_labels']

    # Proses analisis (cache di session state)
    if st.session_state['df_results'] is None:
        with st.spinner(f"Memproses {len(texts)} kalimat..."):
            df_res = analyze_texts(texts, nb_model, svm_model, vec)
            st.session_state['df_results'] = df_res
    else:
        df_res = st.session_state['df_results']

    if df_res.empty:
        st.warning("Tidak ada segmen valid. Coba kalimat yang lebih spesifik.")
        st.stop()

    # ── STATISTIK RINGKAS ─────────────────────────────────────
    total_seg    = len(df_res)
    pos_svm      = (df_res['Prediksi SVM'] == 'Positif').sum()
    neg_svm      = (df_res['Prediksi SVM'] == 'Negatif').sum()
    pos_nb       = (df_res['Prediksi NB'] == 'Positif').sum()
    neg_nb       = (df_res['Prediksi NB'] == 'Negatif').sum()
    unique_texts = df_res['Teks Asli'].nunique()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{unique_texts}</div>
            <div class="metric-label">Kalimat Diuji</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_seg}</div>
            <div class="metric-label">Total Segmen</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:#3FB950;">{pos_svm}</div>
            <div class="metric-label">Positif (SVM)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:#F85149;">{neg_svm}</div>
            <div class="metric-label">Negatif (SVM)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:#3FB950;">{pos_nb}</div>
            <div class="metric-label">Positif (NB)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:#F85149;">{neg_nb}</div>
            <div class="metric-label">Negatif (NB)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABEL HASIL ───────────────────────────────────────────
    st.markdown("### 📋 Hasil Analisis Per Segmen")

    def badge(label):
        cls = 'positif' if label == 'Positif' else 'negatif'
        icon = '●'
        return f'<span class="badge badge-{cls}">{icon} {label}</span>'

    def aspek_badge(asp):
        if asp == 'Lainnya':
            return f'<span class="badge badge-lainnya">— Lainnya</span>'
        return f'<span style="font-size:0.82rem;color:#8B949E;">{asp}</span>'

    rows_html = ""
    for _, row in df_res.iterrows():
        rows_html += f"""
        <tr>
            <td style="max-width:200px;"><span class="mono">{row['Segmen Bersih']}</span></td>
            <td>{aspek_badge(row['Aspek'])}</td>
            <td>{badge(row['Prediksi SVM'])}</td>
            <td>{badge(row['Prediksi NB'])}</td>
            <td><span class="mono" style="font-size:0.75rem;color:#6E7681;">{row['Probabilitas NB']}</span></td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#1C2128;border:1px solid #30363D;border-radius:10px;overflow:hidden;margin-bottom:20px;">
        <table class="result-table">
            <thead><tr>
                <th>Segmen Bersih</th>
                <th>Aspek</th>
                <th>Prediksi SVM</th>
                <th>Prediksi NB</th>
                <th>Probabilitas NB</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── VISUALISASI ───────────────────────────────────────────
    st.markdown("### 📊 Visualisasi Distribusi")

    tab1, tab2, tab3 = st.tabs(["🥧 Distribusi Sentimen", "📊 Per Aspek", "☁️ Word Cloud"])

    with tab1:
        fig_pie = plot_distribusi_pie(df_res)
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with tab2:
        fig_bar = plot_aspek_bar(df_res)
        if fig_bar:
            st.pyplot(fig_bar)
            plt.close(fig_bar)
        else:
            st.info("Tidak ada segmen dengan aspek teridentifikasi (semua 'Lainnya').")

    with tab3:
        fig_wc = plot_wordcloud(df_res)
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    # ── EVALUASI AKURASI (jika ada label) ─────────────────────
    if has_labels and any(l is not None for l in labels):
        st.divider()
        st.markdown("### 🎯 Evaluasi Akurasi")

        # Mapping prediksi ke kalimat asli (segmen pertama per kalimat)
        eval_data = []
        for i, text in enumerate(texts):
            if labels[i] is None: continue
            rows_t = df_res[df_res['Teks Asli'] == text]
            if rows_t.empty: continue
            ps = rows_t.iloc[0]['Prediksi SVM']
            pn = rows_t.iloc[0]['Prediksi NB']
            eval_data.append({
                'Teks'            : text[:70] + '...' if len(text) > 70 else text,
                'Label Sebenarnya': labels[i],
                'Prediksi SVM'    : ps,
                'Prediksi NB'     : pn,
                'SVM'             : '✅' if ps == labels[i] else '❌',
                'NB'              : '✅' if pn == labels[i] else '❌',
            })

        if eval_data:
            df_eval = pd.DataFrame(eval_data)
            y_true  = df_eval['Label Sebenarnya']
            y_svm   = df_eval['Prediksi SVM']
            y_nb    = df_eval['Prediksi NB']

            acc_svm  = accuracy_score(y_true, y_svm)
            acc_nb   = accuracy_score(y_true, y_nb)
            f1_svm   = f1_score(y_true, y_svm, average='weighted', zero_division=0)
            f1_nb    = f1_score(y_true, y_nb,  average='weighted', zero_division=0)

            # Metrik 4 kolom
            m1, m2, m3, m4 = st.columns(4)
            for col_m, val, lbl, sub in [
                (m1, f'{acc_svm:.1%}', 'Akurasi SVM',  'LinearSVC'),
                (m2, f'{acc_nb:.1%}',  'Akurasi NB',   'Multinomial NB'),
                (m3, f'{f1_svm:.1%}',  'F1-Score SVM', 'weighted'),
                (m4, f'{f1_nb:.1%}',   'F1-Score NB',  'weighted'),
            ]:
                with col_m:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-model">{sub}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabel detail benar/salah
            st.markdown("**Detail Prediksi per Kalimat**")
            detail_html = "".join([f"""
            <tr>
                <td style="max-width:280px;font-size:0.82rem;">{r['Teks']}</td>
                <td>{badge(r['Label Sebenarnya'])}</td>
                <td>{badge(r['Prediksi SVM'])}</td>
                <td>{badge(r['Prediksi NB'])}</td>
                <td style="text-align:center;font-size:1rem;">{r['SVM']}</td>
                <td style="text-align:center;font-size:1rem;">{r['NB']}</td>
            </tr>""" for _, r in df_eval.iterrows()])

            st.markdown(f"""
            <div style="background:#1C2128;border:1px solid #30363D;border-radius:10px;overflow:hidden;margin-bottom:20px;">
                <table class="result-table">
                    <thead><tr>
                        <th>Teks</th><th>Label Asli</th>
                        <th>Prediksi SVM</th><th>Prediksi NB</th>
                        <th style="text-align:center;">SVM ✓</th>
                        <th style="text-align:center;">NB ✓</th>
                    </tr></thead>
                    <tbody>{detail_html}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # Chart akurasi + confusion matrix
            col_chart1, col_chart2 = st.columns([1, 2])
            with col_chart1:
                fig_acc = plot_akurasi_bar(acc_svm, acc_nb)
                st.pyplot(fig_acc)
                plt.close(fig_acc)
            with col_chart2:
                labels_cm = sorted(y_true.unique())
                fig_cm    = plot_confusion_matrix(y_true, y_svm, y_nb, labels_cm)
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            # Download hasil evaluasi
            st.download_button(
                "📥 Download Hasil Evaluasi (CSV)",
                data=df_eval.to_csv(index=False).encode('utf-8'),
                file_name="evaluasi_realtime.csv",
                mime="text/csv"
            )

    # ── DOWNLOAD HASIL LENGKAP ────────────────────────────────
    st.divider()
    st.download_button(
        "📥 Download Semua Hasil Analisis (CSV)",
        data=df_res.to_csv(index=False).encode('utf-8'),
        file_name="hasil_analisis_realtime.csv",
        mime="text/csv",
        use_container_width=False
    )
