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
KONJUNGSI = r'\b(tetapi|namun|dan|karena|meskipun|tapi|sedangkan)\b'
KONJUNGSI_SET = {'tetapi', 'namun', 'dan', 'karena', 'meskipun', 'tapi', 'sedangkan'}

TFIDF_PARAMS = {
    'max_features': 3000,
    'ngram_range': (1, 2),
    'sublinear_tf': True,
    'min_df': 2,
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
    score = 0
    for i, word in enumerate(words):
        if word in lexicon:
            val = lexicon[word]
            if i > 0 and words[i-1] in negation_words:
                val = -val
            elif i > 1 and words[i-2] in negation_words:
                val = -val
            score += val
    if score > 0: return 'Positif'
    elif score < 0: return 'Negatif'
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
    st.header("2. Preprocessing & Segmentasi")
    st.warning("1 baris tweet dapat dipecah menjadi beberapa segmen berdasarkan konjungsi.")

    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw'].copy()
        col_name = st.selectbox("Pilih kolom teks:", df.columns)

        if st.button("Mulai Preprocessing (Semua Data)"):
            with st.spinner("Cleaning → Normalisasi → Segmentasi → Stopword → Stemming ..."):
                my_bar = st.progress(0)

                for percent_complete in range(100):
                    time.sleep(0.1) # Simulasi proses
                    my_bar.progress(percent_complete + 1)
                records = []
                df['doc_id'] = df.index
                df['segmen_list'] = df[col_name].apply(preprocess_text)
                df_exploded = df.explode('segmen_list').dropna(subset=['segmen_list'])
                df_exploded = df_exploded.rename(columns={'segmen_list': 'segment'})
                df_exploded = df_exploded[df_exploded['segment'].str.strip() != ''].reset_index(drop=True)
                st.session_state['df_exploded'] = df_exploded
                st.session_state['preprocessing_done'] = True

            st.success(f"Selesai! **{len(df)}** dokumen → **{len(df_exploded)}** segmen.")
            st.dataframe(df_exploded[['doc_id', col_name, 'segment']].head(10))

        if st.session_state['preprocessing_done']:
            st.divider()
            csv_data = st.session_state['df_exploded'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil Preprocessing (CSV)",
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

            st.dataframe(df[['segment', 'sentiment_label', 'aspect_list']].head(10))

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
            wc1, wc2, wc3 = st.columns(3)
            for col_wc, label_wc, cmap_wc in [
                (wc1, 'Positif', 'Greens'),
                (wc2, 'Negatif', 'Reds'),
                (wc3, 'Netral', 'Blues'),
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

        remove_neutral = st.checkbox("Hapus data Netral dari training?", value=True)
        df_model = df_exp[df_exp['sentiment_label'] != 'Netral'].copy() if remove_neutral else df_exp.copy()

        if st.button("Mulai Training Model"):
            with st.spinner("Ekstraksi TF-IDF dan persiapan data..."):
                X = df_model['segment']
                y = df_model['sentiment_label']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                tfidf = TfidfVectorizer(**TFIDF_PARAMS)
                X_train_vec = tfidf.fit_transform(X_train)
                X_test_vec = tfidf.transform(X_test)
                
            # --- ANIMASI PROGRESS BAR (MIMIC APLIKASI2) ---
            col_prog1, col_prog2 = st.columns(2)
            
            with col_prog1:
                st.write("**Melatih Naive Bayes...**")
                pb_nb = st.progress(0)
                t_nb = time.perf_counter()
                nb = MultinomialNB()
                
                # Simulasi progres agar UI tidak kaku
                for i in range(50):
                    time.sleep(0.005)
                    pb_nb.progress(i + 1)
                    
                nb.fit(X_train_vec, y_train)
                pb_nb.progress(100)
                t_nb = time.perf_counter() - t_nb
                y_pred_nb = nb.predict(X_test_vec)
                
            with col_prog2:
                st.write("**Melatih LinearSVC...**")
                pb_svm = st.progress(0)
                t_svm = time.perf_counter()
                svm = LinearSVC(class_weight='balanced', random_state=42)
                
                # Simulasi progres agar UI tidak kaku
                for i in range(50):
                    time.sleep(0.005)
                    pb_svm.progress(i + 1)
                    
                svm.fit(X_train_vec, y_train)
                pb_svm.progress(100)
                t_svm = time.perf_counter() - t_svm
                y_pred_svm = svm.predict(X_test_vec)

            # Simpan ke session state
            st.session_state['model_nb'] = nb
            st.session_state['model_svm'] = svm
            st.session_state['vectorizer'] = tfidf
            st.session_state['y_test'] = y_test
            st.session_state['y_pred_nb'] = y_pred_nb
            st.session_state['y_pred_svm'] = y_pred_svm
            st.session_state['t_nb'] = t_nb
            st.session_state['t_svm'] = t_svm
            
            st.session_state['model_nb_final'] = nb
            st.session_state['model_svm_final'] = svm
            st.session_state['vectorizer_final'] = tfidf

            test_df = df_model.loc[X_test.index].copy()
            test_df['y_true'] = y_test.values
            test_df['pred_nb'] = y_pred_nb
            test_df['pred_svm'] = y_pred_svm
            st.session_state['test_data_eval'] = test_df

            # Simpan model ke disk
            saved = {
                'model_nb': nb, 'model_svm': svm, 'vectorizer': tfidf,
                'y_test': y_test, 'y_pred_nb': y_pred_nb, 'y_pred_svm': y_pred_svm,
                'test_data_eval': test_df, 't_nb': t_nb, 't_svm': t_svm
            }
            try:
                joblib.dump(saved, 'saved_model_data.joblib')
            except Exception as e:
                st.warning(f"Training selesai, namun gagal simpan ke disk: {e}")



        # ── TAMPILKAN HASIL EVALUASI SIDE-BY-SIDE ──────────
        if 'y_test' in st.session_state:
            y_test = st.session_state['y_test']
            y_pred_nb = st.session_state['y_pred_nb']
            y_pred_svm = st.session_state['y_pred_svm']
            t_nb = st.session_state.get('t_nb', 0.0)
            t_svm = st.session_state.get('t_svm', 0.0)

            st.subheader("Matriks Evaluasi Global (Data Uji 20%)")
            
            col_eval1, col_eval2 = st.columns(2)
            labels_cm = sorted(pd.concat([pd.Series(y_test), pd.Series(y_pred_nb), pd.Series(y_pred_svm)]).unique())

            with col_eval1:
                # 1. Tabel Metrik Naive Bayes
                metrics_nb = {
                    "model": "Naive Bayes",
                    "accuracy": accuracy_score(y_test, y_pred_nb),
                    "precision": precision_score(y_test, y_pred_nb, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred_nb, average='weighted', zero_division=0),
                    "f1": f1_score(y_test, y_pred_nb, average='weighted', zero_division=0),
                    "train_time (s)": round(t_nb, 4)
                }
                st.dataframe(pd.DataFrame([metrics_nb]).set_index("model").style.format("{:.4f}"), use_container_width=True)
                
                # 2. Confusion Matrix Naive Bayes
                fig_nb, ax_nb = plt.subplots(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_test, y_pred_nb, labels=labels_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels_cm, yticklabels=labels_cm, ax=ax_nb)
                ax_nb.set_title("Confusion Matrix Naive Bayes", fontsize=14)
                ax_nb.set_xlabel("Predicted")
                ax_nb.set_ylabel("Actual")
                st.pyplot(fig_nb)
                plt.close(fig_nb)

                # 3. MENGGUNAKAN ST.CODE AGAR RAPIH
                with st.expander("Detail Classification Report (NB)"):
                    st.code(classification_report(y_test, y_pred_nb, zero_division=0))

            with col_eval2:
                # 1. Tabel Metrik LinearSVC
                metrics_svm = {
                    "model": "LinearSVC",
                    "accuracy": accuracy_score(y_test, y_pred_svm),
                    "precision": precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
                    "f1": f1_score(y_test, y_pred_svm, average='weighted', zero_division=0),
                    "train_time (s)": round(t_svm, 4)
                }
                st.dataframe(pd.DataFrame([metrics_svm]).set_index("model").style.format("{:.4f}"), use_container_width=True)
                
                # 2. Confusion Matrix LinearSVC
                fig_svm, ax_svm = plt.subplots(figsize=(6, 5))
                sns.heatmap(confusion_matrix(y_test, y_pred_svm, labels=labels_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels_cm, yticklabels=labels_cm, ax=ax_svm)
                ax_svm.set_title("Confusion Matrix LinearSVC", fontsize=14)
                ax_svm.set_xlabel("Predicted")
                ax_svm.set_ylabel("Actual")
                st.pyplot(fig_svm)
                plt.close(fig_svm)

                # 3. MENGGUNAKAN ST.CODE AGAR RAPIH
                with st.expander("Detail Classification Report (SVM)"):
                    st.code(classification_report(y_test, y_pred_svm, zero_division=0))

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

            st.button("Lanjut ke Evaluasi Per Aspek →", on_click=set_page, args=(PAGES[4],))

    else:
        st.warning("Lakukan Pelabelan di Tab 3 terlebih dahulu.")
        
# ============================================================
# TAB 5: EVALUASI DETAIL PER ASPEK
# ============================================================
elif menu == PAGES[4]:
    st.header("5. Evaluasi Detail Per Aspek")

    if 'test_data_eval' in st.session_state:
        df_eval = st.session_state['test_data_eval']

        # ── Matriks Global ────────────────────────────────────────────────────
        st.subheader("Evaluasi Global (Data Uji 20%)")

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

        # Bar chart global
        fig_g, ax_g = plt.subplots(figsize=(8, 6))
        df_global.reset_index().melt(id_vars='Model', var_name='Matriks', value_name='Skor').pipe(
            lambda d: sns.barplot(data=d, x='Matriks', y='Skor', hue='Model', palette='viridis', ax=ax_g)
        )
        ax_g.set_ylim(0, 1.1)
        ax_g.set_title("Perbandingan Matriks Global NB vs SVM")
        for p in ax_g.patches:
            if p.get_height() > 0:
                ax_g.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width()/2, p.get_height()),
                              ha='center', va='bottom', fontsize=9)
        st.pyplot(fig_g)
        plt.close(fig_g)

        st.divider()

        # ── Evaluasi Per Aspek ───────────────────────────────────────────────
        st.subheader("Evaluasi Per Aspek")
        df_exp_eval = df_eval.explode('aspect_list') if 'aspect_list' in df_eval.columns else df_eval.copy()

        asp_col = 'aspect_list' if 'aspect_list' in df_exp_eval.columns else 'aspek'
        aspect_metrics = []
        for asp in df_exp_eval[asp_col].unique():
            sub = df_exp_eval[df_exp_eval[asp_col] == asp]
            if len(sub) > 0:
                aspect_metrics.append({
                    'Aspek': asp,
                    'Jumlah Segmen Uji': len(sub),
                    'Akurasi NB': accuracy_score(sub['y_true'], sub['pred_nb']),
                    'F1 NB': f1_score(sub['y_true'], sub['pred_nb'], average='weighted', zero_division=0),
                    'Akurasi SVM': accuracy_score(sub['y_true'], sub['pred_svm']),
                    'F1 SVM': f1_score(sub['y_true'], sub['pred_svm'], average='weighted', zero_division=0),
                })

        df_asp_met = pd.DataFrame(aspect_metrics).sort_values('Jumlah Segmen Uji', ascending=False)
        fmt_cols = {c: '{:.2%}' for c in df_asp_met.columns if 'Akurasi' in c or 'F1' in c}
        st.dataframe(df_asp_met.style.format(fmt_cols))

        # Bar chart per aspek
        fig_asp, ax_asp = plt.subplots(figsize=(10, 5))
        df_asp_met.melt(id_vars=['Aspek'], value_vars=['Akurasi NB', 'Akurasi SVM'],
                        var_name='Model', value_name='Akurasi').pipe(
            lambda d: sns.barplot(data=d, x='Aspek', y='Akurasi', hue='Model', palette='coolwarm', ax=ax_asp)
        )
        ax_asp.set_ylim(0, 1.1)
        ax_asp.set_title("Perbandingan Akurasi NB vs SVM per Aspek")
        plt.xticks(rotation=30)
        for p in ax_asp.patches:
            if p.get_height() > 0:
                ax_asp.annotate(f'{p.get_height():.2f}',
                                (p.get_x() + p.get_width()/2, p.get_height()),
                                ha='center', va='bottom', fontsize=8)
        st.pyplot(fig_asp)
        plt.close(fig_asp)

        # ── WordCloud Per Aspek ──────────────────────────────────────────────
        st.divider()
        st.subheader("WordCloud per Aspek")
        aspek_list = df_asp_met['Aspek'].tolist()
        if aspek_list:
            tabs_wc = st.tabs(aspek_list)
            for i, asp in enumerate(aspek_list):
                with tabs_wc[i]:
                    sub_asp = df_exp_eval[df_exp_eval[asp_col] == asp]
                    seg_col = 'segment' if 'segment' in sub_asp.columns else 'segment'
                    wc1, wc2, wc3 = st.columns(3)
                    for col_w, label_w, cmap_w in [
                        (wc1, 'Positif', 'Greens'),
                        (wc2, 'Negatif', 'Reds'),
                        (wc3, 'Netral', 'Blues'),
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
    else:
        st.warning("Latih model di Tab 4 terlebih dahulu.")
