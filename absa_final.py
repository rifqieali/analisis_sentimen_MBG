"""
ABSA MBG - Analisis Sentimen Program Makan Bergizi Gratis
Gabungan terbaik dari App1 dan App3:
- Pipeline App1: stemming Sastrawi, SVC(kernel='linear'), fitur NB diagnostik (prior, likelihood, bobot SVM)
- Optimasi App3: LinearSVC, TF-IDF terbatas, struktur ringan, CSS elegan
- TF-IDF: max_features=3000, ngram=(1,2), sublinear_tf=True (dari App3)
- Model SVM: LinearSVC(class_weight='balanced') — lebih cepat dari SVC(kernel='linear') dan akurasi setara
- Tanpa tab Prediksi Manual (dihapus sesuai permintaan)
"""

import time

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
import os
import warnings
from io import BytesIO

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score,
    f1_score, confusion_matrix
)
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisis Sentimen MBG", layout="wide", page_icon="📑")

# ============================================================
# RESOURCE LOADING (CACHED)
# ============================================================
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()

    factory_stop = StopWordRemoverFactory()
    sw_sastrawi = set(factory_stop.get_stop_words())
    sw_custom = {
        "yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp",
        "biar", "bikin", "udah", "udh", "aja", "sih", "deh", "nih",
        "lah", "dong", "kan", "tuh", "mah", "wkwk", "haha", "hehe",
        "aku", "saya", "kamu", "dia", "kita", "kami", "mereka", "sama"
    }
    negation_words = {
        'tidak', 'tak', 'tiada', 'bukan', 'jangan',
        'belum', 'kurang', 'gak', 'ga', 'nggak', 'enggak'
    }
    final_stopwords = (sw_sastrawi | sw_custom) - negation_words
    return stemmer, final_stopwords, negation_words

stemmer, final_stopwords, negation_words = load_resources()

@st.cache_data
def load_normalization_dict():
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        xls = pd.read_excel(BytesIO(resp.content), engine="openpyxl")
        cols = xls.columns.tolist()
        return dict(zip(
            xls[cols[0]].astype(str).str.lower(),
            xls[cols[1]].astype(str).str.lower()
        ))
    except Exception:
        return {"yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "bgt": "banget"}

norm_dict = load_normalization_dict()

@st.cache_data
def load_inset_lexicon():
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    try:
        df_pos = pd.read_csv(pos_url, sep='\t', names=['word', 'weight'], header=None)
        df_neg = pd.read_csv(neg_url, sep='\t', names=['word', 'weight'], header=None)
        df_lex = pd.concat([df_pos, df_neg], ignore_index=True)
        df_lex['weight'] = pd.to_numeric(df_lex['weight'], errors='coerce')
        df_lex = df_lex.dropna(subset=['weight'])
        df_lex['word'] = df_lex['word'].astype(str).str.strip()
        df_lex['weight'] = df_lex['weight'].astype(int)
        lexicon = dict(zip(df_lex['word'], df_lex['weight']))
        # Hapus kata topik netral agar tidak bias skor sentimen
        for w in {'mbg', 'makan', 'bergizi', 'gratis', 'gizi', 'program',
                  'prabowo', 'jokowi', 'presiden', 'menteri', 'indonesia'}:
            lexicon.pop(w, None)
        return lexicon
    except Exception:
        return {}

lexicon = load_inset_lexicon()

# ============================================================
# KONSTANTA
# ============================================================
DOMAIN_LEXICON = {
    # ── KUALITAS ──────────────────────────────────────────
    # Tidak ada di InSet
    "aroma"       :  2,
    "aromanya"    :  2,   # form dengan sufiks
    "lauk"        :  1,   # netral-positif, konteksnya tergantung modifier
    "higienis"    :  3,
    "higienitas"  :  3,
    "keracunan"   : -5,
    "mentah"      : -3,
    "matang"      :  2,
    "bergizi"     :  3,
    "bernutrisi"  :  3,
    "lezat"       :  5,   # sudah ada, tapi eksplisitkan ulang
    "tidak layak" : -4,   # bigram — ditangani terpisah
    "basi"        : -4,
    "ulat"        : -5,
    "porsi"       :  1,   # netral, tergantung modifier
    # ── LAYANAN ───────────────────────────────────────────
    "terlambat"   : -3,
    "molor"       : -3,
    "katering"    :  1,
    "vendor"      :  1,
    "terdistribusi":  2,
    "pelosok"     :  1,   # netral, jadi bermakna jika "sampai pelosok"
    "merata"      :  3,   # naikkan dari +1
    "tepat waktu" :  3,   # bigram
    "tidak merata": -4,   # bigram
    # ── ANGGARAN ──────────────────────────────────────────
    "markup"      : -4,
    "disunat"     : -4,
    "dikorupsi"   : -5,
    "diselewengkan": -5,
    "tepat sasaran":  3,  # bigram
    "tidak tepat sasaran": -4,
}

BIGRAM_OVERRIDE = {
    # Konteks positif yang kata dasarnya negatif di InSet
    "bau enak"      :  2,
    "bau harum"     :  3,
    "bau wangi"     :  3,
    "bau sedap"     :  2,
    # Bigram negatif penting
    "tidak merata"  : -4,
    "tidak layak"   : -4,
    "tidak higienis": -4,
    "tidak sehat"   : -3,
    "tidak tepat"   : -2,
    "kurang enak"   : -2,
    "kurang bersih" : -2,
    "kurang merata" : -3,
    "tepat sasaran" :  3,
    "tepat waktu"   :  3,
}

ABSOLUTE_NEGATIF = {'korupsi', 'keracunan', 'basi', 'ulat', 'markup', 'sunat', 'bocor', 'hambar', 'keras', 'mentah'}
ABSOLUTE_POSITIF = {'terbantu', 'kenyang', 'sehat', 'lezat', 'merata'}

# Gabungkan: domain lexicon override InSet untuk kata yang konflik
final_lexicon = {**lexicon, **DOMAIN_LEXICON}

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
KONJUNGSI = r'\b(tetapi|namun|meskipun|tapi|sedangkan|cuman|cuma|sayangnya|padahal|walau|walaupun|pasalnya)\b'
KONJUNGSI_SET = {'tetapi', 'namun', 'meskipun', 'tapi', 'sedangkan', 'cuman', 'cuma', 'sayangnya', 'padahal', 'walau', 'walaupun', 'pasalnya'}

TFIDF_PARAMS = {
    'max_features': 3000,
    'ngram_range': (1, 2),
    'sublinear_tf': True,
    'min_df': 2,
    'max_df': 0.85
}

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
    cleaned = clean_text(text)
    normed = normalize_text(cleaned)
    segments = segmentasi_kalimat(normed)
    result = []
    for seg in segments:
        processed = stopword_and_stem(seg)
        if processed.strip():
            result.append(processed)
    return result

def get_aspects(text: str) -> list:
    tokens = set(str(text).split())
    found = [asp for asp, keys in ASPEK_DICT.items() if not tokens.isdisjoint(keys)]
    return found if found else ['Lainnya']

def determine_sentiment(text: str) -> str:
    if not isinstance(text, str):
        return 'Netral'

    words = text.split()
    tokens_set = set(words)
    score = 0

    # ── Helper: cek apakah kata di posisi i dinegasi ──────────
    def is_negated(i):
        # Cek 1-3 kata sebelumnya
        for j in range(max(0, i - 3), i):
            if words[j] in negation_words:
                return True
        return False

    # ── 1. ABSOLUTE NEGATIF (dengan cek negasi) ───────────────
    for i, word in enumerate(words):
        if word in ABSOLUTE_NEGATIF:
            if not is_negated(i):
                return 'Negatif'   # hanya jika tidak dinegasi

    # ── 2. ABSOLUTE POSITIF (dengan cek negasi) ───────────────
    for i, word in enumerate(words):
        if word in ABSOLUTE_POSITIF:
            if not is_negated(i):
                return 'Positif'

    # ── 3. BIGRAM OVERRIDE ────────────────────────────────────
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i+1]
        if bigram in BIGRAM_OVERRIDE:
            score += BIGRAM_OVERRIDE[bigram]

    # ── 4. UNIGRAM LEXICON ────────────────────────────────────
    for i, word in enumerate(words):
        if i > 0 and (words[i-1] + " " + word) in BIGRAM_OVERRIDE:
            continue
        if word in final_lexicon:
            val = final_lexicon[word]
            if is_negated(i):
                val = -val
            score += val

    if score >= 0:
        return 'Positif'
    return 'Negatif'

# ============================================================
# SESSION STATE & NAVIGASI
# ============================================================
PAGES = [
    "Upload Data",
    "Preprocessing & Segmentasi",
    "Labeling & Aspek",
    "Modeling (Training)",
    "Evaluasi Detail (Per Aspek)",
    "Pengujian Real-Time"
]

for key, default in [
    ('current_page', PAGES[0]),
    ('df_raw', None),
    ('df_exploded', None),
    ('preprocessing_done', False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def set_page(page_name):
    st.session_state['current_page'] = page_name

st.title("Analisis Sentimen Program Makan Bergizi Gratis")
st.markdown("Berbasis **InSet Lexicon**, **Stemming Sastrawi**, dan **Machine Learning** (MultinomialNB vs LinearSVC)")

menu = st.sidebar.selectbox("Pilih Tahapan", PAGES, key='current_page')

# ============================================================
# TAB 1: UPLOAD DATA
# ============================================================
if menu == PAGES[0]:
    st.header("Upload Dataset CSV")

    data_type = st.radio(
        "Jenis data yang diupload:",
        ("Data Mentah (Belum Preprocessing)", "Data Hasil Preprocessing (Lewati ke Labeling)")
    )

    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if data_type == "Data Mentah (Belum Preprocessing)":
            st.session_state['df_raw'] = df
            st.session_state['df_exploded'] = None
            st.session_state['preprocessing_done'] = False
            st.success(f"Data mentah dimuat: **{len(df)}** baris.")
            st.dataframe(df.head())
            st.button("Lanjut ke Preprocessing →", on_click=set_page, args=(PAGES[1],))
        else:
            if 'segment' not in df.columns:
                st.error("Kolom 'segment' tidak ditemukan. Pastikan file hasil preprocessing memiliki kolom 'segment'.")
            else:
                st.session_state['df_exploded'] = df
                st.session_state['df_raw'] = None
                st.session_state['preprocessing_done'] = True
                st.success(f"Data preprocessing dimuat: **{len(df)}** segmen.")
                st.dataframe(df.head())
                st.button("Lanjut ke Labeling →", on_click=set_page, args=(PAGES[2],))

# ============================================================
# TAB 2: PREPROCESSING & SEGMENTASI
# ============================================================
elif menu == PAGES[1]:
    st.header("Preprocessing & Segmentasi")
    st.warning("1 baris tweet dapat dipecah menjadi beberapa segmen berdasarkan konjungsi.")

    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw'].copy()
        col_name = st.selectbox("Pilih kolom teks:", df.columns)

        if st.button("Mulai Preprocessing (Semua Data)"):
            with st.spinner("Menjalankan pipeline preprocessing bertahap..."):
                count_raw = len(df)
                
                # --- TAHAP 0: FILTER ANTI-BUZZER (Template Match) ---
                df = df.drop_duplicates(subset=[col_name], keep='first').copy()
                df['doc_id'] = range(len(df)) # Reset ulang nomor urut ID
                count_awal = len(df)

                # 1. Cleaning & Case Folding
                df['text_clean'] = df[col_name].apply(clean_text)
                count_clean = len(df)

                # 2. Tokenisasi & Normalisasi
                df['text_norm'] = df['text_clean'].apply(normalize_text)
                count_norm = len(df)

                # 3. Segmentasi (Memecah list menjadi baris baru / explode)
                df['segmen_list'] = df['text_norm'].apply(segmentasi_kalimat)
                df_exploded = df.explode('segmen_list').dropna(subset=['segmen_list'])
                df_exploded['segment'] = df_exploded['segmen_list'].astype(str).str.strip()
                df_exploded = df_exploded[df_exploded['segment'] != '']
                count_segmentasi = len(df_exploded)

                # 4 & 5. Stopword Removal & Stemming (Progress Bar Real-time)
                my_bar = st.progress(0)
                total_rows = count_segmentasi
                
                processed_segments = []
                for i, seg in enumerate(df_exploded['segment']):
                    no_stop = " ".join([w for w in seg.split() if w not in final_stopwords])
                    stemmed = stemmer.stem(no_stop) if no_stop.strip() else ""
                    processed_segments.append(stemmed)
                    
                    if i % max(1, total_rows // 100) == 0:
                        my_bar.progress(min(i / total_rows, 1.0))
                my_bar.progress(1.0)
                
                df_exploded['segment'] = processed_segments
                df_exploded = df_exploded[df_exploded['segment'].str.strip() != ''].reset_index(drop=True)
                count_final = len(df_exploded)

                st.session_state['df_exploded'] = df_exploded
                st.session_state['preprocessing_done'] = True

                # Simpan metrik ke memori
                st.session_state['prep_stats'] = {
                    "raw": count_raw, "awal": count_awal, "clean": count_clean, 
                    "norm": count_norm, "segmentasi": count_segmentasi, "final": count_final
                }

        # --- TAMPILKAN STATISTIK DATA (RINGKAS & TIDAK BESAR) ---
        if st.session_state.get('preprocessing_done', False):
            stats = st.session_state.get('prep_stats', {})
            if stats:
                st.success("Pipeline Preprocessing selesai!")
                
                st.markdown(f"""
                **Statistik Perubahan Jumlah Data:**
                * **Data Mentah Awal:** `{stats['raw']}` baris
                * **Setelah Filter Duplikat (Anti-Buzzer):** `{stats['awal']}` baris *(Dibuang {stats['raw'] - stats['awal']} baris)*
                * **Setelah Cleaning & Normalisasi:** `{stats['norm']}` baris
                * **Setelah Segmentasi (Pecah Konjungsi):** `{stats['segmentasi']}` segmen *(Bertambah {stats['segmentasi'] - stats['norm']} pecahan)*
                * **Final (Setelah Stopword & Stemming):** `{stats['final']}` segmen bersih *(Dibuang {stats['segmentasi'] - stats['final']} segmen kosong/tak bermakna)*
                """)

            # --- TAMPILAN TRACE TEXT (BEFORE-AFTER PER TAHAPAN) ---
            st.divider()
            st.markdown("### Jejak Perubahan Teks Secara Rinci (Before-After)")
            st.caption("Membedah transformasi kalimat langkah demi langkah sesuai urutan pipeline NLP (diambil 3 sampel pertama).")
            
            sample_df = df.head(3)
            for idx, row in sample_df.iterrows():
                raw_text = str(row[col_name])
                
                # 1. Cleaning
                c_text = re.sub(r'@[A-Za-z0-9_]+', ' ', raw_text)
                c_text = re.sub(r'#\w+', ' ', c_text)
                c_text = re.sub(r'http\S+|www\S+', ' ', c_text)
                c_text = re.sub(r'\d+', ' ', c_text)
                c_text = c_text.translate(str.maketrans('', '', string.punctuation))
                c_text = re.sub(r'\s+', ' ', c_text).strip()
                
                # 2. Case Folding
                cf_text = c_text.lower()
                
                # 3. Tokenization
                tok_list = cf_text.split()
                
                # 4. Normalization
                norm_list = [norm_dict.get(w, w) for w in tok_list]
                n_text = " ".join(norm_list)
                
                # 5. Segmentasi
                s_list = segmentasi_kalimat(n_text)
                
                # 6 & 7. Stopword & Stemming
                stopword_results = []
                stem_results = []
                
                for s in s_list:
                    no_stopword = " ".join([w for w in s.split() if w not in final_stopwords])
                    if no_stopword.strip():
                        stopword_results.append(no_stopword)
                        stemmed = stemmer.stem(no_stopword)
                        if stemmed.strip():
                            stem_results.append(stemmed)
                
                with st.expander(f"Sampel {idx+1}: {raw_text[:60]}..."):
                    st.markdown(f"**0. Teks Asli:**<br> `{raw_text}`", unsafe_allow_html=True)
                    st.markdown(f"**1. Cleaning (Hapus Simbol/Angka):**<br> `{c_text}`", unsafe_allow_html=True)
                    st.markdown(f"**2. Case Folding (Huruf Kecil):**<br> `{cf_text}`", unsafe_allow_html=True)
                    st.markdown(f"**3. Tokenization (Pemecahan Array):**<br> `{tok_list}`", unsafe_allow_html=True)
                    st.markdown(f"**4. Normalization (Kata Baku):**<br> `{n_text}`", unsafe_allow_html=True)
                    
                    seg_str = "".join([f"<li><code>{s}</code></li>" for s in s_list])
                    st.markdown(f"**5. Segmentasi (Pecah Konjungsi):**<ul>{seg_str}</ul>", unsafe_allow_html=True)
                    
                    stop_str = "".join([f"<li><code>{s}</code></li>" for s in stopword_results])
                    st.markdown(f"**6. Stopword Removal:**<ul>{stop_str}</ul>", unsafe_allow_html=True)
                    
                    fin_str = "".join([f"<li><code>{f}</code></li>" for f in stem_results])
                    st.markdown(f"**7. Stemming Sastrawi:**<ul>{fin_str}</ul>", unsafe_allow_html=True)

            st.divider()
            st.markdown("**Preview Hasil Data Akhir:**")
            st.dataframe(st.session_state['df_exploded'][['doc_id', col_name, 'segment']].head(10), use_container_width=True)

            csv_data = st.session_state['df_exploded'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Hasil Preprocessing (CSV)",
                data=csv_data,
                file_name="hasil_preprocessing.csv",
                mime="text/csv"
            )
            st.button("Lanjut ke Labeling →", on_click=set_page, args=(PAGES[2],))
    else:
        st.warning("Upload data mentah terlebih dahulu di Tab 1.")

# ============================================================
# TAB 3: LABELING & ASPEK
# ============================================================
elif menu == PAGES[2]:
    st.header("Pelabelan Sentimen & Identifikasi Aspek")

    if st.session_state['df_exploded'] is not None:
        df = st.session_state['df_exploded']

        if st.button("Jalankan Pelabelan & Aspek"):
            with st.spinner("Menentukan sentimen dan aspek per segmen..."):
                df['sentiment_label'] = df['segment'].apply(determine_sentiment)
                df['aspect_list'] = df['segment'].apply(get_aspects)
                st.session_state['df_exploded'] = df

            st.success("Pelabelan selesai!")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Distribusi Sentimen")
                fig, ax = plt.subplots()
                vals = df['sentiment_label'].value_counts()
                ax.pie(vals.values, labels=vals.index, autopct='%1.1f%%')
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                st.subheader("Distribusi Aspek")
                df_asp = df.explode('aspect_list')
                st.bar_chart(df_asp['aspect_list'].value_counts())
            with c3:
                st.subheader("Jumlah Segmen per Sentimen")
                st.bar_chart(df['sentiment_label'].value_counts())

            # Contoh per aspek
            st.subheader("Contoh Segmen per Aspek")
            df_asp2 = df.explode('aspect_list')
            unique_asp = df_asp2['aspect_list'].dropna().unique()
            if len(unique_asp) > 0:
                tabs_asp = st.tabs([str(a) for a in unique_asp])
                for i, asp in enumerate(unique_asp):
                    with tabs_asp[i]:
                        subset = df_asp2[df_asp2['aspect_list'] == asp][['segment', 'sentiment_label']].head(5)
                        st.table(subset.reset_index(drop=True))

            # WordCloud
            st.divider()
            st.subheader("Visualisasi WordCloud")
            wc1, wc2 = st.columns(2)
            for col_wc, label_wc, cmap_wc in [
                (wc1, 'Positif', 'Greens'),
                (wc2, 'Negatif', 'Reds'),
            ]:
                with col_wc:
                    st.markdown(f"**{label_wc}**")
                    text_wc = " ".join(df[df['sentiment_label'] == label_wc]['segment'].astype(str))
                    if text_wc.strip():
                        wc = WordCloud(width=400, height=300, background_color='white', colormap=cmap_wc).generate(text_wc)
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                        plt.close(fig_wc)

        if 'sentiment_label' in st.session_state['df_exploded'].columns:
            st.button("Lanjut ke Modeling →", on_click=set_page, args=(PAGES[3],))
    else:
        st.warning("Lakukan Preprocessing terlebih dahulu.")

# ============================================================
# TAB 4: MODELING (TRAINING)
# ============================================================
elif menu == PAGES[3]:
    st.header("Training Model (MultinomialNB vs LinearSVC)")

    df_exp = st.session_state.get('df_exploded')
    if df_exp is not None and 'sentiment_label' in df_exp.columns:

        # --- AUTO-LOAD MODEL SILENTLY (JIKA ADA) ---
        if 'model_nb' not in st.session_state and os.path.exists('saved_model_data.joblib'):
            try:
                saved = joblib.load('saved_model_data.joblib')
                for k in ['model_nb', 'model_svm', 'vectorizer', 'y_test',
                          'y_pred_nb', 'y_pred_svm', 'test_data_eval', 't_nb', 't_svm']:
                    if k in saved:
                        st.session_state[k] = saved[k]
                        if k in ['model_nb', 'model_svm', 'vectorizer']:
                            st.session_state[f"{k}_final"] = saved[k]
            except Exception:
                pass

        df_model = df_exp.copy()

        if st.button("Mulai Training Model"):
            with st.spinner("Mengeksekusi 3 Skenario Pengujian (70:30, 80:20, 90:10) secara paralel..."):
                X = df_model['segment']
                y = df_model['sentiment_label']

                skenario_splits = {
                    "70:30": 0.3,
                    "80:20": 0.2,
                    "90:10": 0.1
                }
                
                hasil_skenario = {}
                pb = st.progress(0)
                progress_step = 0

                for name, test_size in skenario_splits.items():
                    # 1. Split Data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )

                    # 2. Ekstraksi TF-IDF
                    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
                    X_train_vec = tfidf.fit_transform(X_train)
                    X_test_vec = tfidf.transform(X_test)
                    
                    # 3. Latih Naive Bayes (Multinomial standar krn data sudah 51:49)
                    t_nb = time.perf_counter()
                    nb = MultinomialNB(alpha=0.1) 
                    nb.fit(X_train_vec, y_train)
                    t_nb = time.perf_counter() - t_nb
                    y_pred_nb = nb.predict(X_test_vec)
                    
                    progress_step += 15
                    pb.progress(progress_step)
                    
                    # 4. Latih LinearSVC
                    t_svm = time.perf_counter()
                    svm = LinearSVC(random_state=42)
                    svm.fit(X_train_vec, y_train)
                    t_svm = time.perf_counter() - t_svm
                    y_pred_svm = svm.predict(X_test_vec)
                    
                    progress_step += 15
                    pb.progress(progress_step)
                    
                    # 5. Simpan Hasil per Skenario
                    test_df = df_model.loc[X_test.index].copy()
                    test_df['y_true'] = y_test.values
                    test_df['pred_nb'] = y_pred_nb
                    test_df['pred_svm'] = y_pred_svm
                    
                    hasil_skenario[name] = {
                        'model_nb': nb, 'model_svm': svm, 'vectorizer': tfidf,
                        'y_test': y_test, 'y_pred_nb': y_pred_nb, 'y_pred_svm': y_pred_svm,
                        't_nb': t_nb, 't_svm': t_svm, 'test_data_eval': test_df
                    }
                    
                    # 6. Tetapkan 80:20 sebagai Model Final untuk Tab 6
                    if name == "80:20":
                        st.session_state['model_nb'] = nb
                        st.session_state['model_svm'] = svm
                        st.session_state['vectorizer'] = tfidf
                        st.session_state['model_nb_final'] = nb
                        st.session_state['model_svm_final'] = svm
                        st.session_state['vectorizer_final'] = tfidf
                        st.session_state['test_data_eval'] = test_df # Eval tab 5 pakai yg 80:20
                        
                pb.progress(100)
                st.session_state['hasil_skenario'] = hasil_skenario
                st.success("Training untuk ketiga skenario selesai!")

        # ── TAMPILKAN HASIL EVALUASI DENGAN TABS ──────────
        if 'hasil_skenario' in st.session_state:
            st.divider()
            st.subheader("Matriks Evaluasi Global Berdasarkan Skenario Split")
            
            # Membuat 3 Tab
            tab70, tab80, tab90 = st.tabs(["**70:30**", "**80:20**", "**90:10**"])
            
            tabs_dict = {"70:30": tab70, "80:20": tab80, "90:10": tab90}
            
            for split_name, tab in tabs_dict.items():
                with tab:
                    data = st.session_state['hasil_skenario'][split_name]
                    y_t = data['y_test']
                    p_nb = data['y_pred_nb']
                    p_svm = data['y_pred_svm']
                    
                    col_eval1, col_eval2 = st.columns(2)
                    labels_cm = sorted(pd.concat([pd.Series(y_t), pd.Series(p_nb), pd.Series(p_svm)]).unique())

                    with col_eval1:
                        metrics_nb = {
                            "model": "Multinomial NB",
                            "accuracy": accuracy_score(y_t, p_nb),
                            "precision": precision_score(y_t, p_nb, average='weighted', zero_division=0),
                            "recall": recall_score(y_t, p_nb, average='weighted', zero_division=0),
                            "f1": f1_score(y_t, p_nb, average='weighted', zero_division=0),
                            "train_time (s)": round(data['t_nb'], 4)
                        }
                        st.dataframe(pd.DataFrame([metrics_nb]).set_index("model").style.format("{:.4f}"), use_container_width=True)
                        
                        fig_nb, ax_nb = plt.subplots(figsize=(5, 4))
                        sns.heatmap(confusion_matrix(y_t, p_nb, labels=labels_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels_cm, yticklabels=labels_cm, ax=ax_nb)
                        ax_nb.set_title(f"Confusion Matrix MultinomialNB ({split_name})", fontsize=12)
                        st.pyplot(fig_nb)
                        plt.close(fig_nb)

                        with st.expander("Detail Classification Report (NB)"):
                            st.code(classification_report(y_t, p_nb, zero_division=0))

                    with col_eval2:
                        metrics_svm = {
                            "model": "LinearSVC",
                            "accuracy": accuracy_score(y_t, p_svm),
                            "precision": precision_score(y_t, p_svm, average='weighted', zero_division=0),
                            "recall": recall_score(y_t, p_svm, average='weighted', zero_division=0),
                            "f1": f1_score(y_t, p_svm, average='weighted', zero_division=0),
                            "train_time (s)": round(data['t_svm'], 4)
                        }
                        st.dataframe(pd.DataFrame([metrics_svm]).set_index("model").style.format("{:.4f}"), use_container_width=True)
                        
                        fig_svm, ax_svm = plt.subplots(figsize=(5, 4))
                        sns.heatmap(confusion_matrix(y_t, p_svm, labels=labels_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels_cm, yticklabels=labels_cm, ax=ax_svm)
                        ax_svm.set_title(f"Confusion Matrix LinearSVC ({split_name})", fontsize=12)
                        st.pyplot(fig_svm)
                        plt.close(fig_svm)

                        with st.expander("Detail Classification Report (SVM)"):
                            st.code(classification_report(y_t, p_svm, zero_division=0))

            # ── Visualisasi TF-IDF top words ────────────────────────────────
            st.divider()
            st.subheader("Top 10 Kata Bobot TF-IDF Tertinggi")
            tfidf_vec = st.session_state['vectorizer']
            X_train_for_viz = tfidf_vec.transform(st.session_state['test_data_eval']['segment'])
            sum_tfidf = X_train_for_viz.sum(axis=0)
            words_arr = tfidf_vec.get_feature_names_out()
            df_tfidf = pd.DataFrame({
                'Kata': words_arr,
                'Skor TF-IDF': np.asarray(sum_tfidf).flatten()
            }).sort_values('Skor TF-IDF', ascending=False).head(10).reset_index(drop=True)

            c_tf1, c_tf2 = st.columns([1, 2])
            with c_tf1:
                st.dataframe(df_tfidf, use_container_width=True)
            with c_tf2:
                fig_tf, ax_tf = plt.subplots(figsize=(8, 5))
                sns.barplot(x='Skor TF-IDF', y='Kata', data=df_tfidf, palette='viridis',
                            hue='Kata', legend=False, ax=ax_tf)
                ax_tf.set_title("Top 10 Kata TF-IDF (Data Uji)")
                st.pyplot(fig_tf)
                plt.close(fig_tf)

            st.button("Lanjut ke Evaluasi Per Aspek →", on_click=set_page, args=(PAGES[4],))

    else:
        st.warning("Lakukan Pelabelan di Tab 3 terlebih dahulu.")
        
# ============================================================
# TAB 5: EVALUASI DETAIL PER ASPEK
# ============================================================
elif menu == PAGES[4]:
    st.header("Evaluasi Detail Per Aspek")

    if 'hasil_skenario' in st.session_state:
        # Pilihan Skenario untuk Evaluasi Detail
        skenario_options = list(st.session_state['hasil_skenario'].keys())
        selected_scenario = st.selectbox(
            "Pilih Skenario Split Data untuk Dilihat Detailnya:",
            options=skenario_options,
            index=1  # Default ke 80:20
        )
        
        # Ambil data dari skenario yang dipilih
        data_eval = st.session_state['hasil_skenario'][selected_scenario]
        df_eval = data_eval['test_data_eval']

        # ── Evaluasi Per Aspek ───────────────────────────────────────────────
        df_exp_eval = df_eval.explode('aspect_list')
        
        aspect_metrics = []
        unique_aspects = [a for a in df_exp_eval['aspect_list'].unique() if pd.notna(a)]
        
        for asp in unique_aspects:
            sub = df_exp_eval[df_exp_eval['aspect_list'] == asp]
            if len(sub) > 0:
                aspect_metrics.append({
                    'Aspek': asp,
                    'Jumlah Data Uji': len(sub),
                    'Akurasi NB': accuracy_score(sub['y_true'], sub['pred_nb']),
                    'F1 NB': f1_score(sub['y_true'], sub['pred_nb'], average='weighted', zero_division=0),
                    'Akurasi SVM': accuracy_score(sub['y_true'], sub['pred_svm']),
                    'F1 SVM': f1_score(sub['y_true'], sub['pred_svm'], average='weighted', zero_division=0),
                })

        df_asp_met = pd.DataFrame(aspect_metrics)
        
        if not df_asp_met.empty:
            df_asp_met = df_asp_met.sort_values('Jumlah Data Uji', ascending=False)
            
            # Tampilkan tabel metrik per aspek
            fmt_cols = {c: '{:.2%}' for c in df_asp_met.columns if 'Akurasi' in c or 'F1' in c}
            st.dataframe(df_asp_met.style.format(fmt_cols), use_container_width=True)

            # Visualisasi Perbandingan Akurasi Per Aspek
            fig_asp, ax_asp = plt.subplots(figsize=(10, 5))
            df_plot = df_asp_met.melt(id_vars=['Aspek'], value_vars=['Akurasi NB', 'Akurasi SVM'],
                                     var_name='Model', value_name='Akurasi')
            
            sns.barplot(data=df_plot, x='Aspek', y='Akurasi', hue='Model', palette='coolwarm', ax=ax_asp)
            
            ax_asp.set_ylim(0, 1.1)
            ax_asp.set_title(f"Perbandingan Akurasi NB vs SVM per Aspek ({selected_scenario})")
            plt.xticks(rotation=30)
            
            # Tambahkan label angka di atas bar
            for p in ax_asp.patches:
                if p.get_height() > 0:
                    ax_asp.annotate(f'{p.get_height():.2%}', 
                                    (p.get_x() + p.get_width()/2, p.get_height()), 
                                    ha='center', va='bottom', fontsize=8)
            
            st.pyplot(fig_asp)
            plt.close(fig_asp)
        else:
            st.info("Tidak ada data aspek untuk ditampilkan (Data metrik kosong).")

        # ── WordCloud Per Aspek ──────────────────────────────────────────────
        st.divider()
        st.subheader("WordCloud per Aspek")
        aspek_list = df_asp_met['Aspek'].tolist() if not df_asp_met.empty else []
        if aspek_list:
            tabs_wc = st.tabs(aspek_list)
            for i, asp in enumerate(aspek_list):
                with tabs_wc[i]:
                    sub_asp = df_exp_eval[df_exp_eval['aspect_list'] == asp]
                    seg_col = 'segment' if 'segment' in sub_asp.columns else 'segment'
                    wc1, wc2 = st.columns(2)
                    for col_w, label_w, cmap_w in [
                        (wc1, 'Positif', 'Greens'),
                        (wc2, 'Negatif', 'Reds'),
                    ]:
                        with col_w:
                            st.markdown(f"##### {label_w}")
                            text_w = " ".join(sub_asp[sub_asp['y_true'] == label_w][seg_col].astype(str))
                            if text_w.strip():
                                wc_obj = WordCloud(width=300, height=200, background_color='white',
                                                   colormap=cmap_w).generate(text_w)
                                fig_wc, ax_wc = plt.subplots()
                                ax_wc.imshow(wc_obj, interpolation='bilinear')
                                ax_wc.axis('off')
                                st.pyplot(fig_wc)
                                plt.close(fig_wc)
                            else:
                                st.caption(f"Tidak ada data {label_w.lower()}.")

            st.button("Pengujian Real-Time", on_click=set_page, args=(PAGES[5],))


    else:
        st.warning("Latih model di Tab 4 terlebih dahulu.")
# ============================================================
# TAB 6: PENGUJIAN REAL-TIME
# ============================================================
elif menu == PAGES[5]:
    st.header("Pengujian Model Real-Time")
    
    if 'model_nb' not in st.session_state or 'model_svm' not in st.session_state:
        st.warning("Anda belum melatih model. Silakan kembali ke Tab Modeling dan klik 'Mulai Training'.")
    else:
        
        user_input = st.text_area("Masukkan opini atau teks baru terkait Program MBG:", 
                                  height=150, 
                                  placeholder="")
        
        if st.button("Analisis Teks"):
            if user_input.strip() == "":
                st.error("Teks tidak boleh kosong.")
            else:
                with st.spinner("Memproses pipeline NLP (Cleaning, Segmentasi, Stopword, Stemming)..."):
                    # 1. Masuk ke Pipeline Preprocessing yang sama dengan Tab 2
                    processed_segments = preprocess_text(user_input)
                    
                    if not processed_segments:
                        st.warning("Teks tidak menghasilkan kata yang bermakna setelah melewati filter *stopword*.")
                    else:
                        # 2. Ambil model dan vectorizer dari memori
                        nb_model = st.session_state['model_nb']
                        svm_model = st.session_state['model_svm']
                        vec = st.session_state['vectorizer']
                        
                        # 3. Transformasi ke TF-IDF
                        X_input_tfidf = vec.transform(processed_segments)
                        
                        # 4. Lakukan Prediksi
                        preds_nb = nb_model.predict(X_input_tfidf)
                        preds_svm = svm_model.predict(X_input_tfidf)
                        probs_nb = nb_model.predict_proba(X_input_tfidf)
                        classes = nb_model.classes_
                        
                        # 5. Siapkan output tabel
                        st.divider()
                        st.subheader("Hasil Analisis Per Segmen")
                        
                        def get_color(label):
                            if str(label) == 'Positif': return "🟢 Positif"
                            return "🔴 Negatif"
                        
                        results_data = []
                        for i, seg in enumerate(processed_segments):
                            # Ambil aspek menggunakan fungsi dari Tab 3
                            aspect_list = get_aspects(seg)
                            aspect_str = ", ".join(aspect_list)
                            
                            # Format persentase probabilitas Naive Bayes
                            prob_str = " | ".join([f"{cls}: {probs_nb[i][j]:.1%}" for j, cls in enumerate(classes)])
                            
                            results_data.append({
                                "Segmen Teks Bersih": seg,
                                "Aspek Terdeteksi": aspect_str,
                                "Prediksi LinearSVC": get_color(preds_svm[i]),
                                "Prediksi Naive Bayes": get_color(preds_nb[i]),
                                "Probabilitas NB": prob_str
                            })
                        
                        st.table(pd.DataFrame(results_data))
                        
                