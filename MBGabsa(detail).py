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
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
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
def load_roberta():
    local_model_path = "./roberta_sentiment_local"
    MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

    if not os.path.exists(local_model_path):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)

    return pipeline(
        "text-classification",
        model=local_model_path,
        tokenizer=local_model_path,
        device=-1,
        truncation=True,
        max_length=128
    )
classifier = load_roberta()

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

def determine_sentiment_roberta(text: str, classifier) -> str:
    if not isinstance(text, str) or not text.strip():
        return 'Netral'
    try:
        result = classifier(text[:512])[0]
        label = result['label'].lower()
        if 'pos' in label:
            return 'Positif'
        elif 'neg' in label:
            return 'Negatif'
        return 'Netral'
    except Exception:
        return 'Netral'

# ============================================================
# SESSION STATE & NAVIGASI
# ============================================================
PAGES = [
    "1. Upload Data",
    "2. Preprocessing & Segmentasi",
    "3. Labeling & Aspek",
    "4. Modeling (Training)",
    "5. Evaluasi Detail (Per Aspek)",
    "6. PENGUJIAN REAL-TIME",
]

for key, default in [
    ('current_page', PAGES[0]),
    ('df_raw', None),
    ('df_exploded', None),
    ('preprocessing_done', False),
    ('labeling_done', False),
    ('df_neutral_handled', None),
    ('neutral_action', None),
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
    st.header("1. Upload Dataset CSV")

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
            st.session_state['labeling_done'] = False
            st.session_state['df_neutral_handled'] = None
            st.session_state['neutral_action'] = None
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
                st.session_state['labeling_done'] = False
                st.session_state['df_neutral_handled'] = None
                st.session_state['neutral_action'] = None
                st.success(f"Data preprocessing dimuat: **{len(df)}** segmen.")
                st.dataframe(df.head())
                st.button("Lanjut ke Labeling →", on_click=set_page, args=(PAGES[2],))

# ============================================================
# TAB 2: PREPROCESSING & SEGMENTASI
# ============================================================
elif menu == PAGES[1]:
    st.header("2. Preprocessing & Segmentasi")
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
                st.session_state['labeling_done'] = False
                st.session_state['df_neutral_handled'] = None
                st.session_state['neutral_action'] = None

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
    st.header("3. Pelabelan Sentimen & Identifikasi Aspek")

    if st.session_state['df_exploded'] is not None:
        df = st.session_state['df_exploded']

        st.markdown("### ⚙️ Pengaturan Labeling Biner")
        handle_neutral = st.radio(
            "Pilih penanganan untuk opini yang terdeteksi 'Netral' oleh RoBERTa:",
            ("Hapus Data Netral (Hanya simpan Positif & Negatif)", "Ubah data Netral menjadi Positif")
        )

        if not st.session_state['labeling_done']:
            if st.button("Jalankan Pelabelan & Aspek"):
                with st.spinner("Menentukan sentimen dan aspek per segmen..."):
                    classifier = load_roberta()
                    
                    # Progress bar untuk pelabelan RoBERTa
                    total_rows = len(df)
                    my_bar = st.progress(0)
                    
                    sentiments = []
                    for i, seg in enumerate(df['segment']):
                        sentiments.append(determine_sentiment_roberta(seg, classifier))
                        # Update progress bar secara berkala agar UI tidak kaku
                        if i % max(1, total_rows // 100) == 0:
                            my_bar.progress(min((i + 1) / total_rows, 1.0))
                    my_bar.progress(1.0)
                    
                    df['sentiment_label'] = sentiments
                    df['aspect_list'] = df['segment'].apply(get_aspects)

                    # Simpan data netral sebelum diubah/dihapus untuk ditampilkan
                    df_neutral = df[df['sentiment_label'] == 'Netral'].copy()
                    st.session_state['df_neutral_handled'] = df_neutral
                    st.session_state['neutral_action'] = handle_neutral

                    # Menangani label Netral sesuai pilihan pengguna
                    if handle_neutral == "Hapus Data Netral (Hanya simpan Positif & Negatif)":
                        df = df[df['sentiment_label'] != 'Netral'].reset_index(drop=True)
                    else:
                        df['sentiment_label'] = df['sentiment_label'].replace('Netral', 'Positif')

                    st.session_state['df_exploded'] = df
                    st.session_state['labeling_done'] = True
                    st.rerun()
        else:
            df = st.session_state['df_exploded']
            st.success("Pelabelan selesai!")

            if st.button("🔄 Ulangi Pelabelan"):
                st.session_state['labeling_done'] = False
                st.session_state['df_neutral_handled'] = None
                st.session_state['neutral_action'] = None
                if 'sentiment_label' in st.session_state['df_exploded'].columns:
                    st.session_state['df_exploded'].drop(columns=['sentiment_label', 'aspect_list'], inplace=True, errors='ignore')
                st.rerun()

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

            st.dataframe(df[['segment', 'sentiment_label', 'aspect_list']].head(10))

            # --- Tampilkan Info Data Netral ---
            if st.session_state.get('df_neutral_handled') is not None and not st.session_state['df_neutral_handled'].empty:
                st.divider()
                st.subheader("⚠️ Data Sentimen Netral yang Ditangani")
                action_text = "dihapus dari dataset" if "Hapus" in st.session_state['neutral_action'] else "diubah menjadi Positif"
                st.info(f"Terdapat **{len(st.session_state['df_neutral_handled'])}** segmen yang awalnya terdeteksi **Netral** oleh model RoBERTa dan telah **{action_text}** sesuai pengaturan Anda.")
                
                st.dataframe(st.session_state['df_neutral_handled'][['segment', 'sentiment_label', 'aspect_list']].head(10))

            # Contoh per aspek
            st.divider()
            st.subheader("Contoh Segmen per Aspek")
            df_asp2 = df.explode('aspect_list')
            unique_asp = df_asp2['aspect_list'].dropna().unique()
            if len(unique_asp) > 0:
                tabs_asp = st.tabs([str(a) for a in unique_asp])
                for i, asp in enumerate(unique_asp):
                    with tabs_asp[i]:
                        subset = df_asp2[df_asp2['aspect_list'] == asp][['segment', 'sentiment_label']].head(5)
                        st.table(subset.reset_index(drop=True))

            # Frekuensi kata per sentimen
            st.divider()
            st.subheader("Top 10 Kata per Sentimen")
            sentiments = df['sentiment_label'].unique()
            cols_freq = st.columns(len(sentiments))
            for idx, sent in enumerate(sentiments):
                with cols_freq[idx]:
                    st.markdown(f"**{sent}**")
                    all_words = " ".join(df[df['sentiment_label'] == sent]['segment'].astype(str)).split()
                    if all_words:
                        freq = pd.Series(all_words).value_counts().head(10)
                        fig_f, ax_f = plt.subplots(figsize=(5, 5))
                        sns.barplot(x=freq.values, y=freq.index, ax=ax_f, palette='viridis', hue=freq.index, legend=False)
                        ax_f.set_xlabel("Frekuensi")
                        st.pyplot(fig_f)
                        plt.close(fig_f)

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
    st.header("4. Training Model (NB vs LinearSVC)")

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
                    nb = MultinomialNB() 
                    nb.fit(X_train_vec, y_train)
                    t_nb = time.perf_counter() - t_nb
                    y_pred_nb = nb.predict(X_test_vec)
                    
                    progress_step += 15
                    pb.progress(progress_step)
                    
                    # 4. Latih LinearSVC
                    t_svm = time.perf_counter()
                    svm = LinearSVC()
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
                st.success("✅ Training untuk ketiga skenario selesai!")

        # ── TAMPILKAN HASIL EVALUASI DENGAN TABS ──────────
        if 'hasil_skenario' in st.session_state:
            st.divider()
            st.subheader("Matriks Evaluasi Global Berdasarkan Skenario Split")
            
            # Membuat 3 Tab
            tab70, tab80, tab90 = st.tabs(["📊 Skenario 70:30", "⭐ Skenario 80:20 (Golden Ratio)", "📈 Skenario 90:10"])
            
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
                        ax_nb.set_title(f"Confusion Matrix NB ({split_name})", fontsize=12)
                        ax_nb.set_xlabel("Prediksi Model")
                        ax_nb.set_ylabel("Label Aktual")
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
                        ax_svm.set_xlabel("Prediksi Model")
                        ax_svm.set_ylabel("Label Aktual")
                        st.pyplot(fig_svm)
                        plt.close(fig_svm)

                        with st.expander("Detail Classification Report (SVM)"):
                            st.code(classification_report(y_t, p_svm, zero_division=0))

            # ── Visualisasi TF-IDF top words (Berdasarkan Model 80:20) ────────────────────────────────
            st.divider()
            # (Lanjutan kode TF-IDF dan Bedah Diagnostik tetap sama persis seperti sebelumnya)
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
                
            # ── Bedah Perhitungan Matematis TF-IDF ──────────────────────────
            st.divider()
            st.subheader("Bedah Perhitungan Matematis TF-IDF (Teks Uji)")
            st.info("💡 **FAKTA UNTUK SIDANG SKRIPSI:** Scikit-Learn tidak menghitung 'TF × IDF' secara mentah. Karena Anda menyetel `sublinear_tf=True`, rumusnya berubah menjadi **(1 + log(TF)) × IDF**. Setelah itu, hasilnya dinormalisasi (L2 Norm) agar model tidak bias terhadap kalimat yang terlalu panjang.")
            
            contoh_teks = st.text_input("Masukkan teks contoh untuk membedah perhitungan mesin:", value="dukung program makan gizi gratis mantap")
            
            if contoh_teks:
                vec = st.session_state['vectorizer']
                idf_dict = dict(zip(vec.get_feature_names_out(), vec.idf_))
                
                # Mendapatkan N_train (Estimasi jumlah dokumen latih dari 80:20 split)
                N_train = len(st.session_state['test_data_eval']) * 4
                
                # Tokenisasi sederhana
                tokens = contoh_teks.lower().split()
                from collections import Counter
                tf_raw = Counter(tokens)
                
                table_data = []
                bobot_mentah_list = []
                
                for kata, count in tf_raw.items():
                    if kata in idf_dict:
                        idf_val = idf_dict[kata]
                        # Reverse-engineer nilai DF aktual dari rumus smooth_idf
                        # idf = ln((N+1)/(DF+1)) + 1  --> DF = (N+1)/exp(idf-1) - 1
                        df_val = int(round((N_train + 1) / np.exp(idf_val - 1) - 1))
                        
                        # Rumus Sublinear TF: 1 + log(tf)
                        tf_sublinear = 1 + np.log(count)
                        
                        # TF-IDF mentah (sebelum L2 Normalization)
                        bobot_mentah = tf_sublinear * idf_val
                        bobot_mentah_list.append(bobot_mentah)
                        
                        table_data.append({
                            "Token Kata": kata,
                            "Nilai TF (Count)": count,
                            "Sublinear TF (1+log)": round(tf_sublinear, 4),
                            "Nilai DF Aktual": df_val,
                            "Perhitungan IDF": round(idf_val, 4),
                            "TF × IDF Mentah": round(bobot_mentah, 4)
                        })
                    else:
                        table_data.append({
                            "Token Kata": kata,
                            "Nilai TF (Count)": count,
                            "Sublinear TF (1+log)": 0,
                            "Nilai DF Aktual": 0,
                            "Perhitungan IDF": 0,
                            "TF × IDF Mentah": 0
                        })
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                
                if bobot_mentah_list:
                    l2_norm = np.sqrt(sum([x**2 for x in bobot_mentah_list]))
                    st.markdown(f"**🔹 Faktor L2 Normalization (Pembagi):** `{round(l2_norm, 4)}`")
                    st.caption(f"*Bobot akhir (Final Weight) yang diekstraksi ke dalam matriks algoritma Naive Bayes & SVM didapatkan dengan membagi setiap kolom 'TF × IDF Mentah' dengan angka {round(l2_norm, 4)} ini.*")

            # ── Diagnostik Naive Bayes: Prior + Likelihood ──────────────────
            st.divider()
            st.subheader("Diagnostik Naive Bayes — Prior & Likelihood")

            nb_model = st.session_state['model_nb']
            tfidf_vec = st.session_state['vectorizer']

            prior_df = pd.DataFrame({
                "Kelas": nb_model.classes_,
                "Prior Probability": np.exp(nb_model.class_log_prior_)
            })
            st.markdown("**Prior Probability (Probabilitas Awal Kelas):**")
            st.dataframe(prior_df.style.format({"Prior Probability": "{:.6f}"}))

            st.markdown("**Likelihood Kata per Kelas:**")
            input_kata = st.text_input("Masukkan kata uji (pisahkan koma):",
                                       value="dukung, makan, gizi, gratis, bagus, jelek", key="nb_kata")
            if input_kata:
                kata_list = [k.strip().lower() for k in input_kata.split(',') if k.strip()]
                lik_data = []
                for kata in kata_list:
                    row = {"Kata": kata}
                    if kata in tfidf_vec.vocabulary_:
                        idx = tfidf_vec.vocabulary_[kata]
                        probs = np.exp(nb_model.feature_log_prob_[:, idx])
                        for i, cls in enumerate(nb_model.classes_):
                            row[f"P(kata|{cls})"] = probs[i]
                    else:
                        for cls in nb_model.classes_:
                            row[f"P(kata|{cls})"] = "tidak ada di vocabulary"
                    lik_data.append(row)
                st.dataframe(pd.DataFrame(lik_data), use_container_width=True)
            
            # ── Diagnostik SVM: Bias + Bobot kata ───────────────────────────
            st.divider()
            st.subheader("Diagnostik LinearSVC — Bias & Bobot Kata")

            svm_model = st.session_state['model_svm']
            tfidf_vec_svm = st.session_state['vectorizer']

            bias_df = pd.DataFrame({
                "Decision Function": [f"Fungsi ke-{i}" for i in range(len(svm_model.intercept_))],
                "Nilai Bias (b)": svm_model.intercept_
            })
            st.markdown("**Nilai Bias (Intercept) SVM:**")
            st.dataframe(bias_df)

            st.markdown("**Bobot Kata (Koefisien w) per Decision Function:**")
            input_kata_svm = st.text_input("Masukkan kata uji SVM (pisahkan koma):",
                                           value="dukung, makan, gizi, gratis, bagus, jelek", key="svm_kata")
            if input_kata_svm:
                kata_svm = [k.strip().lower() for k in input_kata_svm.split(',') if k.strip()]
                coef = np.array(svm_model.coef_)
                svm_data = []
                for kata in kata_svm:
                    row = {"Kata": kata}
                    if kata in tfidf_vec_svm.vocabulary_:
                        idx = tfidf_vec_svm.vocabulary_[kata]
                        for i in range(coef.shape[0]):
                            row[f"w (Fungsi {i})"] = coef[i, idx]
                    else:
                        for i in range(coef.shape[0]):
                            row[f"w (Fungsi {i})"] = "tidak ada di vocabulary"
                    svm_data.append(row)
                st.dataframe(pd.DataFrame(svm_data), use_container_width=True)

            st.button("Lanjut ke Evaluasi Per Aspek", on_click=set_page, args=(PAGES[4],))

    else:
        st.warning("Lakukan Pelabelan di Tab 3 terlebih dahulu.")
        
# ============================================================
# TAB 5: EVALUASI DETAIL PER ASPEK
# ============================================================
elif menu == PAGES[4]:
    st.header("5. Evaluasi Detail Per Aspek")

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

        st.info(f"Menampilkan analisis mendalam untuk Skenario **{selected_scenario}**.")

        # ── 1. Matriks Global Skenario Terpilih ───────────────────────────────
        st.subheader(f"Ringkasan Performa Global ({selected_scenario})")

        def calc_metrics(y_true, y_pred, name):
            return {
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            }

        global_nb = calc_metrics(df_eval['y_true'], df_eval['pred_nb'], "Naive Bayes")
        global_svm = calc_metrics(df_eval['y_true'], df_eval['pred_svm'], "LinearSVC")
        df_global = pd.DataFrame([global_nb, global_svm]).set_index("Model")
        
        st.dataframe(df_global.style.highlight_max(axis=0, color='lightgreen').format("{:.2%}"))

        st.divider()

        # ── 2. Evaluasi Per Aspek ──────────────────────────────────────────────
        st.subheader(f"Evaluasi Per Aspek ({selected_scenario})")
        
        # Explode aspek agar bisa dihitung per kategori
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
        st.warning("Latih model di Tab 4 terlebih dahulu.")
    else:
        # ── INISIALISASI STATE TAB 6 ──────────────────────────
        if 'rt_analyzed' not in st.session_state:
            st.session_state['rt_analyzed'] = False
            st.session_state['rt_texts'] = []
            st.session_state['rt_labels'] = []
            st.session_state['rt_has_labels'] = False

        if not st.session_state['rt_analyzed']:
            mode = st.radio(
                "Mode Input:",
                ("Input Manual (Multi-baris)", "Upload CSV (Batch 50 kalimat)"),
                horizontal=True
            )

            # ── MODE 1: INPUT MANUAL ──────────────────────────────
            if mode == "Input Manual (Multi-baris)":
                st.markdown("**Format:** `kalimat | label` (label: Positif atau Negatif)")
                raw = st.text_area(
                    "Masukkan kalimat uji:",
                    height=300,
                    placeholder="makanan bergizi enak dan porsinya cukup | Positif\ndistribusi sering telat dan tidak merata | Negatif\ndana MBG dikorupsi oknum tidak bertanggung jawab | Negatif",
                    key="rt_manual_input"
                )

                if st.button("Analisis Teks", use_container_width=True):
                    if raw.strip():
                        texts_to_analyze = []
                        true_labels = []
                        has_labels = False
                        for line in raw.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            if '|' in line:
                                parts = line.split('|', 1)
                                texts_to_analyze.append(parts[0].strip())
                                true_labels.append(parts[1].strip().capitalize())
                                has_labels = True
                            else:
                                texts_to_analyze.append(line)
                                true_labels.append(None)
                        
                        st.session_state['rt_texts'] = texts_to_analyze
                        st.session_state['rt_labels'] = true_labels
                        st.session_state['rt_has_labels'] = has_labels
                        st.session_state['rt_analyzed'] = True
                        st.rerun()
                    else:
                        st.warning("Teks uji tidak boleh kosong.")

            # ── MODE 2: UPLOAD CSV ────────────────────────────────
            else:
                st.caption("Format CSV: satu kolom bernama 'full_text', satu baris per kalimat.")

                # Tombol download template
                template_df = pd.DataFrame({"full_text": [
                    "makanan bergizi enak dan porsinya cukup untuk anak sekolah",
                    "distribusi makanan sering telat siswa menunggu berjam-jam",
                    "dana MBG dikorupsi dan dimarkup oknum tidak bertanggung jawab",
                ]})
                st.download_button(
                    "📥 Download Template CSV",
                    data=template_df.to_csv(index=False).encode('utf-8'),
                    file_name="template_pengujian.csv",
                    mime="text/csv"
                )

                uploaded_test = st.file_uploader("Upload CSV kalimat uji:", type="csv")
                
                if st.button("Analisis Semua Kalimat", use_container_width=True):
                    if uploaded_test:
                        df_test = pd.read_csv(uploaded_test)
                        if 'full_text' not in df_test.columns:
                            st.error("Kolom 'full_text' tidak ditemukan di CSV.")
                        else:
                            texts_to_analyze = df_test['full_text'].dropna().astype(str).tolist()
                            st.session_state['rt_texts'] = texts_to_analyze
                            st.session_state['rt_labels'] = [None] * len(texts_to_analyze)
                            st.session_state['rt_has_labels'] = False
                            st.session_state['rt_analyzed'] = True
                            st.rerun()
                    else:
                        st.warning("Silakan upload CSV terlebih dahulu.")
        else:
            st.success("✅ Analisis selesai!")
            if st.button("🔄 Ulangi Analisis", use_container_width=True):
                st.session_state['rt_analyzed'] = False
                st.session_state['rt_texts'] = []
                st.session_state['rt_labels'] = []
                st.session_state['rt_has_labels'] = False
                st.rerun()

        # ── PROSES ANALISIS ───────────────────────────────────
        if st.session_state.get('rt_analyzed') and st.session_state['rt_texts']:
            texts_to_analyze = st.session_state['rt_texts']
            true_labels = st.session_state['rt_labels']
            has_labels = st.session_state['rt_has_labels']

            nb_model = st.session_state['model_nb']
            svm_model = st.session_state['model_svm']
            vec = st.session_state['vectorizer']

            with st.spinner(f"Memproses {len(texts_to_analyze)} kalimat..."):
                results_data = []
                for text in texts_to_analyze:
                    segments = preprocess_text(text)
                    if not segments:
                        continue
                    for seg in segments:
                        X = vec.transform([seg])
                        pred_nb = nb_model.predict(X)[0]
                        pred_svm = svm_model.predict(X)[0]
                        probs_nb = nb_model.predict_proba(X)[0]
                        classes = nb_model.classes_

                        prob_str = " | ".join([
                            f"{cls}: {probs_nb[j]:.1%}"
                            for j, cls in enumerate(classes)
                        ])
                        results_data.append({
                            "Teks Asli": text,
                            "Segmen Bersih": seg,
                            "Aspek": ", ".join(get_aspects(seg)),
                            "Prediksi SVM": pred_svm,
                            "Prediksi NB": pred_nb,
                            "Probabilitas NB": prob_str,
                        })

            if not results_data:
                st.warning("Tidak ada segmen valid yang dihasilkan.")
            else:
                df_results = pd.DataFrame(results_data)
                st.success(f"{len(df_results)} segmen berhasil dianalisis.")

                # ── TABEL HASIL ───────────────────────────────
                st.subheader("Hasil Analisis Per Segmen")
                st.dataframe(df_results, use_container_width=True)

                # ── VISUALISASI ───────────────────────────────
                if has_labels and any(l is not None for l in true_labels):
                    st.divider()
                    st.subheader("Evaluasi Akurasi Pengujian Real-Time")

                    # Mapping hasil prediksi ke label asli per kalimat
                    # (ambil prediksi segmen pertama per kalimat)
                    eval_data = []
                    for i, text in enumerate(texts_to_analyze):
                        if true_labels[i] is None:
                            continue
                        # Ambil prediksi untuk kalimat ini dari df_results
                        rows = df_results[df_results['Teks Asli'] == text]
                        if rows.empty:
                            continue
                        # Ambil prediksi segmen pertama sebagai representasi kalimat
                        pred_svm = rows.iloc[0]['Prediksi SVM']
                        pred_nb = rows.iloc[0]['Prediksi NB']
                        eval_data.append({
                            'Teks': text[:60] + '...' if len(text) > 60 else text,
                            'Label Sebenarnya': true_labels[i],
                            'Prediksi SVM': pred_svm,
                            'Prediksi NB': pred_nb,
                            'SVM Benar': '✅' if pred_svm == true_labels[i] else '❌',
                            'NB Benar': '✅' if pred_nb == true_labels[i] else '❌',
                        })

                    if eval_data:
                        df_eval_rt = pd.DataFrame(eval_data)

                        # Hitung akurasi
                        acc_svm = (df_eval_rt['Prediksi SVM'] == df_eval_rt['Label Sebenarnya']).mean()
                        acc_nb = (df_eval_rt['Prediksi NB'] == df_eval_rt['Label Sebenarnya']).mean()

                        # Metrik ringkas
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Akurasi LinearSVC", f"{acc_svm:.1%}")
                        with col2:
                            st.metric("Akurasi Naive Bayes", f"{acc_nb:.1%}")

                        # Tabel detail benar/salah
                        st.dataframe(df_eval_rt, use_container_width=True)

                        # Bar chart akurasi
                        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
                        models = ['LinearSVC', 'Naive Bayes']
                        accs = [acc_svm, acc_nb]
                        colors = ['#2ecc71' if a >= 0.7 else '#e74c3c' for a in accs]
                        bars = ax_acc.bar(models, accs, color=colors, width=0.4, edgecolor='white')
                        ax_acc.set_ylim(0, 1.1)
                        ax_acc.set_ylabel("Akurasi")
                        ax_acc.set_title("Akurasi Pengujian Real-Time (50 Kalimat)")
                        ax_acc.axhline(y=0.7, color='gray', linestyle='--',
                                    linewidth=1, label='Threshold 70%')
                        ax_acc.legend()
                        for bar, acc in zip(bars, accs):
                            ax_acc.text(
                                bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.02,
                                f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold'
                            )
                        st.pyplot(fig_acc)
                        plt.close(fig_acc)

                        # Confusion matrix per model
                        st.divider()
                        st.markdown("**Confusion Matrix Pengujian Real-Time**")
                        labels_cm = sorted(df_eval_rt['Label Sebenarnya'].unique())
                        fig_cm, axes = plt.subplots(1, 2, figsize=(10, 4))

                        for ax, pred_col, title in [
                            (axes[0], 'Prediksi SVM', 'LinearSVC'),
                            (axes[1], 'Prediksi NB', 'Naive Bayes'),
                        ]:
                            cm = confusion_matrix(
                                df_eval_rt['Label Sebenarnya'],
                                df_eval_rt[pred_col],
                                labels=labels_cm
                            )
                            sns.heatmap(
                                cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels_cm, yticklabels=labels_cm, ax=ax
                            )
                            ax.set_title(f"Confusion Matrix — {title}")
                            ax.set_xlabel("Prediksi")
                            ax.set_ylabel("Aktual")

                        plt.tight_layout()
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                        # Download hasil evaluasi
                        st.download_button(
                            "📥 Download Hasil Evaluasi Real-Time (CSV)",
                            data=df_eval_rt.to_csv(index=False).encode('utf-8'),
                            file_name="evaluasi_realtime.csv",
                            mime="text/csv"
                        )