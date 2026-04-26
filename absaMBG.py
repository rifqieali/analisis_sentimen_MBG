import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import time
import requests
import os
import warnings

from io import BytesIO
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB # Keep MultinomialNB
from sklearn.svm import LinearSVC # Use LinearSVC directly
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold # New import for CV
from sklearn.preprocessing import LabelEncoder # New import for CV
from sklearn.utils.class_weight import compute_class_weight
from wordcloud import WordCloud

#aku ganteng

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen MBG", layout="wide")

# ==========================================
# 1. FUNGSI CACHED
# ==========================================
def load_resources():
    resources = ['punkt']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
    
    factory_stop = StopWordRemoverFactory()
    
    sw_custom = {
        "yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp",
        "biar", "bikin", "udah", "udh", "aja", "sih", "deh", "nih",
        "lah", "dong", "kan", "tuh", "mah", "wkwk", "haha", "hehe",
    }
    negation_words = {'tidak', 'tak', 'tiada', 'bukan', 'jangan', 'belum', 'kurang', 'gak', 'ga', 'nggak', 'enggak'}

    # Menggabungkan stopwords dari Sastrawi (list) dengan stopwords custom (set)
    stopwords_sastrawi = set(factory_stop.get_stop_words())
    final_stopwords = (stopwords_sastrawi | sw_custom) - negation_words

    
    return final_stopwords, negation_words

final_stopwords, negation_words = load_resources()


def load_normalization_dict():
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    try:
        response = requests.get(url)
        file_excel = BytesIO(response.content)
        df_norm = pd.read_excel(file_excel)
        norm_dict = dict(zip(df_norm.iloc[:, 0], df_norm.iloc[:, 1]))
        return norm_dict
    except Exception as e:
        return {"yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "bgt": "banget"}

norm_dict_global = load_normalization_dict()

@st.cache_data
def load_inset_lexicon():
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    try:
        df_pos = pd.read_csv(pos_url, sep='\t', names=['word', 'weight'], header=None)
        df_neg = pd.read_csv(neg_url, sep='\t', names=['word', 'weight'], header=None)
        df_lexicon = pd.concat([df_pos, df_neg], ignore_index=True)
        df_lexicon['weight'] = pd.to_numeric(df_lexicon['weight'], errors='coerce')
        df_lexicon = df_lexicon.dropna(subset=['weight'])
        df_lexicon['word'] = df_lexicon['word'].astype(str).str.strip()
        df_lexicon['weight'] = df_lexicon['weight'].astype(int)
        
        lexicon = dict(zip(df_lexicon['word'], df_lexicon['weight']))
        
        return lexicon
    except Exception:
        return {}

lexicon = load_inset_lexicon()

# Global TF-IDF Parameters
TFIDF_PARAMS = {
    "max_features": 2000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.9,
    "sublinear_tf": True,
}

# --- NEW HELPER FUNCTIONS FOR MODEL TRAINING ---
def extract_tfidf(corpus, fit=True, vectorizer=None):
    """
    Extracts TF-IDF features from a corpus.
    If fit=True, fits a new TfidfVectorizer.
    If fit=False, uses an existing vectorizer to transform.
    """
    if fit:
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        X = tfidf.fit_transform(corpus)
        return X, tfidf
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided if fit is False.")
        X = vectorizer.transform(corpus)
        return X, vectorizer

def train_model_nb(X, y):
    """Trains a Multinomial Naive Bayes model with class balancing."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    prior = weights / weights.sum()
    model = MultinomialNB(class_prior=prior)
    model.fit(X, y)
    return model

def train_model_svm_linear(X, y):
    """Trains a LinearSVC model with class balancing."""
    # Increased max_iter for convergence, probability=True is not needed for LinearSVC
    model = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    model.fit(X, y)
    return model

def cross_validation_pipeline(corpus, labels, model_type, progress_bar=None, n_splits=10):
    """
    Performs N-fold cross-validation for a given model type.
    Returns aggregated true labels and predictions.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_pred_all = np.empty_like(labels.values, dtype=object)
    
    for fold, (train_index, test_index) in enumerate(skf.split(corpus, y_encoded)):
        X_train, X_test = corpus.iloc[train_index], corpus.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
        # TF-IDF Vectorization for each fold
        X_train_tfidf, tfidf_vectorizer = extract_tfidf(X_train, fit=True)
        X_test_tfidf, _ = extract_tfidf(X_test, fit=False, vectorizer=tfidf_vectorizer)
        
        # Model Training
        if model_type == "nb":
            model = train_model_nb(X_train_tfidf, y_train)
        elif model_type == "svm":
            model = train_model_svm_linear(X_train_tfidf, y_train)
        else:
            raise ValueError("model_type must be 'nb' or 'svm'")
            
        y_pred = model.predict(X_test_tfidf)
        
        y_pred_all[test_index] = y_pred
        
        if progress_bar:
            progress_bar.progress((fold + 1) / n_splits)
            
    return {
        "y_true": labels.tolist(),
        "y_pred": y_pred_all.tolist(),
        "model_name": "Naive Bayes" if model_type == "nb" else "LinearSVC"
    }

def evaluate_model(cv_results, model_name):
    """
    Evaluates model performance and generates a confusion matrix.
    Returns metrics, matplotlib figure, and classification report string.
    """
    # ... (Implementation of evaluate_model as provided in the thought process)
    y_true = cv_results["y_true"]
    y_pred = cv_results["y_pred"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Classification Report
    cr = classification_report(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title(f"Confusion Matrix {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    return metrics, fig, cr

lexicon = load_inset_lexicon()

# ==========================================
# 2. FUNGSI UTAMA (Segmentasi & Preprocessing)
# ==========================================
def clean_text(text):
    """Tahap 1 & 2: Cleaning dan Case Folding"""
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    """Tahap 3 & 4: Tokenisasi dan Normalisasi"""
    words = text.split()
    words = [norm_dict_global.get(w, w) for w in words]
    return ' '.join(words)

def segmentasi_kalimat(text):
    """Tahap 5: Segmentasi (Memecah kalimat berdasarkan konjungsi)."""
    konjungsi_list = {'tetapi', 'namun', 'dan', 'karena', 'meskipun', 'tapi', 'sedangkan'}
    konjungsi_regex = r'\b(' + '|'.join(konjungsi_list) + r')\b'
    # Hapus spasi kosong dan filter kata konjungsi itu sendiri agar tidak menjadi segmen terpisah
    segmen = [s.strip() for s in re.split(konjungsi_regex, text) if s and s.strip() and s.strip().lower() not in konjungsi_list]
    return segmen if segmen else [text]

def remove_stopwords(text):
    """Tahap 6: Stopword Removal"""
    words = text.split()
    words = [w for w in words if w not in final_stopwords]
    return ' '.join(words)

def preprocess_text(text):
    """
    Pipeline baru:
    1. Cleaning & Case Folding
    2. Tokenisasi & Normalisasi
    3. Segmentasi
    4. Stopword (per segmen)
    """
    # 1. Cleaning & Case Folding
    text_clean = clean_text(text)
    
    # 2. Tokenisasi & Normalisasi
    text_norm = normalize_text(text_clean)
    
    # 3. Segmentasi
    segmen_list = segmentasi_kalimat(text_norm)
    
    # 4. Stopword (per segmen)
    final_segments = []
    for seg in segmen_list:
        processed = remove_stopwords(seg)
        if processed:
            final_segments.append(processed)
            
    return final_segments

def get_aspects(text):
    aspects = []
    text_words = set(str(text).split())
    keywords = {
        'Kualitas': ["kualitas", "bagus", "jelek", "enak", "basi", "gizi", "susu",
        "menu", "rasa", "porsi", "higienis", "keracunan", "sehat",
        "mentah", "keras", "hambar", "ulat", "lauk", "sayur",
        "karbohidrat", "protein", "lemak", "gula", "ayam", "telur",
        "kenyang", "alergi", "higienitas"],

        'Layanan': ["layan", "antri", "ramah", "lambat", "cepat", "bantu", "saji",
        "distribusi", "vendor", "katering", "sekolah", "siswa", "guru",
        "telat", "molor", "bocor", "tepat waktu", "pelosok", "merata",
        "zonasi", "umkm", "kemasan", "kotak", "plastik"],
        
        'Anggaran': ["harga", "mahal", "murah", "biaya", "bayar", "anggar", "boros",
        "korupsi", "dana", "apbn", "pajak", "potong", "sunat", "markup",
        "tender", "proyek", "apbd", "defisit", "utang", "ekonomi",
        "alokasi", "transparan"]
    }
    for aspect, keys in keywords.items():
        if not text_words.isdisjoint(keys):
            aspects.append(aspect)
    return aspects if aspects else ['Lainnya']

def determine_sentiment(text):
    if not isinstance(text, str): return 'Netral'
    score = 0
    words = text.split()
    for i, word in enumerate(words):
        if word in lexicon:
            val = lexicon[word]
            if i > 0 and words[i-1] in negation_words: val = -val
            elif i > 1 and words[i-2] in negation_words: val = -val
            score += val
    if score > 0: return 'Positif'
    elif score < 0: return 'Negatif'
    else: return 'Netral'

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================
st.title("📊 Analisis Sentimen Program Makan Bergizi Gratis")
st.markdown("Aplikasi berbasis InSet Lexicon & Machine Learning dengan **Segmentasi Konjungsi**")

PAGES = [
    "1. Upload Data", 
    "2. Preprocessing & Segmentasi", 
    "3. Labeling & Aspek", 
    "4. Modeling (Training)", 
    "5. Evaluasi Detail (Per Aspek)",
    "6. Prediksi Manual"
]

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = PAGES[0]

def set_page(page_name):
    st.session_state['current_page'] = page_name

menu = st.sidebar.selectbox("Pilih Tahapan", PAGES, key='current_page')

if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_exploded' not in st.session_state:
    st.session_state['df_exploded'] = None
if 'preprocessing_done' not in st.session_state:   # TAMBAHKAN INI
    st.session_state['preprocessing_done'] = False

# --- TAB 1: UPLOAD ---
if menu == "1. Upload Data":
    st.header("Upload Dataset CSV")
    
    data_type = st.radio(
        "Pilih Jenis Data yang Diupload:",
        ("Data Mentah (Belum Preprocessing)", "Data Hasil Preprocessing (Skip ke Labeling)")
    )
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file:
        # --- PROTOKOL HARD RESET (SAPU JAGAT) ---
        # Begitu file baru masuk, hancurkan SEMUA sisa memori dari proses sebelumnya
        keys_to_destroy = [
            'df_exploded', 'model_nb', 'model_svm', 'y_test', 
            'y_pred_nb', 'y_pred_svm', 'test_data_eval', 'df_tfidf'
        ]
        for key in keys_to_destroy:
            if key in st.session_state:
                del st.session_state[key]
        # ----------------------------------------
        
        df = pd.read_csv(uploaded_file)
        
        if data_type == "Data Mentah (Belum Preprocessing)":
            st.session_state['df_raw'] = df
            st.success(f"✅ Data mentah baru berhasil dimuat! Jumlah baris dokumen: {len(df)}")
            st.info("Sistem telah dibersihkan dari memori lama. Anda siap memproses data baru dari nol.")
            st.dataframe(df.head())
            st.button("Lanjut ke Preprocessing 👉", on_click=set_page, args=(PAGES[1],))
            
        else:
            if 'segment' not in df.columns:
                st.error("❌ File CSV tidak valid! Pastikan terdapat kolom 'segment'.")
            else:
                st.session_state['df_exploded'] = df
                st.session_state['df_raw'] = None # Hapus data mentah jika ada
                st.success(f"✅ Data hasil preprocessing baru berhasil dimuat! Jumlah baris segmen: {len(df)}")
                st.info("Sistem telah dibersihkan dari memori lama. Anda siap melabeli data baru.")
                st.dataframe(df.head())
                st.button("Lanjut ke Labeling 👉", on_click=set_page, args=(PAGES[2],))

# --- TAB 2: PREPROCESSING ---
elif menu == "2. Preprocessing & Segmentasi":
    st.header("Preprocessing & Segmentasi Berbasis Konjungsi")
    st.warning("PERHATIAN: 1 baris teks dapat pecah menjadi beberapa baris segmen berdasarkan konjungsi (dan, tetapi, namun).")
    
    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw'].copy()
        col_name = st.selectbox("Pilih kolom teks yang akan diproses:", df.columns)
        
        # Tombol diperjelas agar Anda tahu ini berfungsi untuk menimpa data (Override)
        if st.button("⚙️ Mulai / Jalankan Ulang Pemrosesan"):
            with st.spinner('Memecah segmen dan membersihkan teks (harap tunggu)...'):
                df['doc_id'] = df.index
                
                # Eksekusi fungsi preprocessing
                df['segmen_list'] = df[col_name].apply(preprocess_text)
                
                df_exploded = df.explode('segmen_list').dropna(subset=['segmen_list'])
                df_exploded.rename(columns={'segmen_list': 'segment'}, inplace=True)
                df_exploded = df_exploded[df_exploded['segment'].str.strip() != '']
                df_exploded = df_exploded.reset_index(drop=True)
                
                # TIMPA MEMORI LAMA dengan data yang baru saja diproses
                st.session_state['df_exploded'] = df_exploded
                
                # Hapus memori Labeling & Modeling jika user melakukan preprocessing ulang
                if 'sentiment_label' in st.session_state['df_exploded'].columns:
                    st.session_state['df_exploded'].drop(columns=['sentiment_label', 'aspect_list'], inplace=True, errors='ignore')
                for key in ['model_nb', 'model_svm', 'y_test', 'test_data_eval']:
                    if key in st.session_state:
                        del st.session_state[key]
        
        # --- FIX UI: TAMPILKAN SELALU JIKA DATA ADA DI MEMORI ---
        # Blok ini diletakkan di luar st.button agar tabel tidak gaib (menghilang) saat refresh
        if st.session_state['df_exploded'] is not None:
            st.success(f"✅ Data berhasil diproses! Menghasilkan total {len(st.session_state['df_exploded'])} segmen kalimat.")
            
            st.write("**Preview Data Hasil Preprocessing:**")
            st.dataframe(st.session_state['df_exploded'][['doc_id', col_name, 'segment']].head(10))
            
            st.divider()
            # --- FITUR DOWNLOAD DATA PREPROCESSING ---
            csv_data = st.session_state['df_exploded'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data Hasil Preprocessing (CSV)",
                data=csv_data,
                file_name="hasil_preprocessing_segmen.csv",
                mime="text/csv"
            )
            st.write("") # Spasi pemisah
            st.button("Lanjut ke Labeling 👉", on_click=set_page, args=(PAGES[2],))
    else:
        st.warning("Upload data mentah terlebih dahulu di Tab 1.")

# --- TAB 3: LABELING ---
elif menu == "3. Labeling & Aspek":
    st.header("Pelabelan & Ekstraksi Aspek (Level Segmen)")
    
    if st.session_state['df_exploded'] is not None:
        df = st.session_state['df_exploded']
        
        if st.button("Jalankan Pelabelan & Aspek"):
            with st.spinner('Menentukan aspek dan sentimen pada setiap segmen...'):
                df['sentiment_label'] = df['segment'].apply(determine_sentiment)
                df['aspect_list'] = df['segment'].apply(get_aspects)
                st.session_state['df_exploded'] = df
            
            st.success("Pelabelan Selesai!")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribusi Sentimen (Level Segmen)")
                st.bar_chart(df['sentiment_label'].value_counts())
            with c2:
                st.subheader("Distribusi Aspek")
                df_asp = df.explode('aspect_list')
                st.bar_chart(df_asp['aspect_list'].value_counts())
                
            st.dataframe(df[['segment', 'sentiment_label', 'aspect_list']].head())
            
            # --- CONTOH DATA PER ASPEK ---
            st.divider()
            st.subheader("Contoh Data Hasil Klasifikasi per Aspek")
            
            # Mengambil daftar aspek unik yang ada di data
            unique_aspects_list = df_asp['aspect_list'].dropna().unique()
            
            if len(unique_aspects_list) > 0:
                # Membuat tab Streamlit untuk tiap aspek agar rapi
                tabs_aspek = st.tabs([str(a) for a in unique_aspects_list])
                for i, aspect in enumerate(unique_aspects_list):
                    with tabs_aspek[i]:
                        st.markdown(f"**5 Contoh Segmen Kalimat (Aspek: {aspect})**")
                        subset_aspek = df_asp[df_asp['aspect_list'] == aspect].head(5) # Ambil minimal 5
                        
                        cols_to_show = ['segment', 'sentiment_label']
                        # Mencari kolom teks asli untuk ditampilkan (kolom selain doc_id dan hasil proses)
                        other_cols = [c for c in subset_aspek.columns if c not in ['doc_id', 'segment', 'sentiment_label', 'aspect_list']]
                        if other_cols:
                            cols_to_show = [other_cols[0]] + cols_to_show
                            
                        st.table(subset_aspek[cols_to_show].reset_index(drop=True))
            
            # --- ANALISIS FREKUENSI KATA ---
            st.divider()
            st.subheader("Analisis Frekuensi Kata Terbanyak per Sentimen")
            
            sentiments_list = df['sentiment_label'].unique()
            cols_freq = st.columns(len(sentiments_list))
            
            for idx, sent in enumerate(sentiments_list):
                with cols_freq[idx]:
                    st.markdown(f"**Top 10 Kata: {sent}**")
                    subset_data = df[df['sentiment_label'] == sent]
                    all_words = " ".join(subset_data['segment'].astype(str)).split()
                    
                    if all_words:
                        freq_data = pd.Series(all_words).value_counts().head(10)
                        fig_freq, ax_freq = plt.subplots(figsize=(5, 6))
                        sns.barplot(x=freq_data.values, y=freq_data.index, ax=ax_freq, palette='viridis')
                        ax_freq.set_xlabel("Frekuensi")
                        st.pyplot(fig_freq)
                        plt.close(fig_freq)
                    else:
                        st.info("Tidak ada data kata.")
            
            # --- VISUALISASI WORDCLOUD ---
            st.divider()
            st.subheader("Visualisasi WordCloud")
            wc_c1, wc_c2, wc_c3 = st.columns(3)
            
            with wc_c1:
                st.markdown("**Sentimen Positif**")
                text_pos = " ".join(df[df['sentiment_label'] == 'Positif']['segment'].astype(str))
                if text_pos.strip():
                    wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(text_pos)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_pos, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
            
            with wc_c2:
                st.markdown("**Sentimen Negatif**")
                text_neg = " ".join(df[df['sentiment_label'] == 'Negatif']['segment'].astype(str))
                if text_neg.strip():
                    wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(text_neg)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_neg, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
            
            with wc_c3:
                st.markdown("**Sentimen Netral**")
                text_neu = " ".join(df[df['sentiment_label'] == 'Netral']['segment'].astype(str))
                if text_neu.strip():
                    wc_neu = WordCloud(width=400, height=300, background_color='white', colormap='Blues').generate(text_neu)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_neu, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
        
        if 'sentiment_label' in df.columns:
            st.button("Lanjut ke Modeling 👉", on_click=set_page, args=(PAGES[3],))
    else:
        st.warning("Lakukan Preprocessing & Segmentasi terlebih dahulu.")

# --- TAB 4: MODELING (TRAINING) ---
elif menu == "4. Modeling (Training)":
    st.header("Training Model Machine Learning")
    
    if st.session_state.get('df_exploded') is not None: # Use .get() for safer access
        df = st.session_state['df_exploded']
        remove_neutral = st.checkbox("Hapus Data Netral?", value=True)
        
        df_model = df[df['sentiment_label'] != 'Netral'] if remove_neutral else df.copy()
            
        if st.button("Mulai Training Model"):
            with st.spinner("Melatih model dengan optimasi tingkat tinggi... (Seharusnya selesai di bawah 2 menit)"):
                
                X = df_model['segment']
                y = df_model['sentiment_label'] # Renamed from labels to y for consistency
                
                col_eval1, col_eval2 = st.columns(2)
                
                # ── Cross Validation NB ──
                with col_eval1:
                    st.write("**Melatih Naive Bayes (10-Fold CV)...**")
                    pb_nb = st.progress(0)
                    t_nb = time.perf_counter()
                    cv_nb_results = cross_validation_pipeline(X, y, model_type="nb", progress_bar=pb_nb)
                    t_nb = time.perf_counter() - t_nb
                    
                    metrics_nb, fig_nb, cr_nb = evaluate_model(cv_nb_results, "Naive Bayes")
                    metrics_nb["train_time"] = round(t_nb, 3)
                    
                    st.dataframe(pd.DataFrame([metrics_nb]).set_index("model").style.format("{:.4f}"))
                    st.pyplot(fig_nb)
                    plt.close(fig_nb)
                    with st.expander("Detail Classification Report (NB)"):
                        st.code(cr_nb)
                
                # ── Cross Validation SVM ──
                with col_eval2:
                    st.write("**Melatih LinearSVC (10-Fold CV)...**")
                    pb_svm = st.progress(0)
                    t_svm = time.perf_counter()
                    cv_svm_results = cross_validation_pipeline(X, y, model_type="svm", progress_bar=pb_svm)
                    t_svm = time.perf_counter() - t_svm
                    
                    metrics_svm, fig_svm, cr_svm = evaluate_model(cv_svm_results, "LinearSVC")
                    metrics_svm["train_time"] = round(t_svm, 3)
                    
                    st.dataframe(pd.DataFrame([metrics_svm]).set_index("model").style.format("{:.4f}"))
                    st.pyplot(fig_svm)
                    plt.close(fig_svm)
                    with st.expander("Detail Classification Report (SVM)"):
                        st.code(cr_svm)
                        
                # ── Latih model final (seluruh data) ──
                with st.spinner("Melatih model final pada seluruh data..."):
                    X_all_tfidf, vectorizer_final = extract_tfidf(X, fit=True)
                    model_nb_final = train_model_nb(X_all_tfidf, y)
                    model_svm_final = train_model_svm_linear(X_all_tfidf, y)
                    
                    # Pilih model terbaik berdasarkan F1-score dari CV
                    best_model = model_svm_final if metrics_svm["f1"] >= metrics_nb["f1"] else model_nb_final
                    best_model_name = "LinearSVC" if metrics_svm["f1"] >= metrics_nb["f1"] else "Naive Bayes"
                    
                    # Simpan ke memori (Session State)
                    st.session_state['best_model'] = best_model
                    st.session_state['best_model_name'] = best_model_name
                    st.session_state['model_nb_final'] = model_nb_final
                    st.session_state['model_svm_final'] = model_svm_final
                    st.session_state['vectorizer_final'] = vectorizer_final
                    
                    # Simpan hasil CV untuk evaluasi detail
                    st.session_state['cv_nb_results'] = cv_nb_results
                    st.session_state['cv_svm_results'] = cv_svm_results
                    st.session_state['df_model_for_eval'] = df_model # Store df_model for aspect evaluation
                    
                st.success(f"Training Selesai! Model terbaik berdasarkan F1-Score: **{best_model_name}**")
        
        st.divider()
        
        if 'best_model' in st.session_state: # Check if a model has been trained
            st.subheader("1. Evaluasi Kinerja Model (Data Testing 20%)")
            
            # Display metrics and confusion matrices from CV results
            metrics_nb_display, fig_nb_display, _ = evaluate_model(st.session_state['cv_nb_results'], "Naive Bayes")
            metrics_svm_display, fig_svm_display, _ = evaluate_model(st.session_state['cv_svm_results'], "LinearSVC")
            
            st.dataframe(pd.DataFrame([metrics_nb_display, metrics_svm_display]).set_index("model").style.highlight_max(axis=0, color='green').format("{:.2%}"))
            
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                st.pyplot(fig_nb_display)
                plt.close(fig_nb_display)
            with col_cm2:
                st.pyplot(fig_svm_display)
                plt.close(fig_svm_display)
            
            # --- TF-IDF VISUALIZATION ---
            st.divider()
            st.subheader("3. Kata dengan Bobot TF-IDF Tertinggi")
            
            try:
                vectorizer = st.session_state['vectorizer_final']
                # Use the full corpus for TF-IDF visualization
                X_full_tfidf, _ = extract_tfidf(df_model['segment'], fit=False, vectorizer=vectorizer)
                
                # Hitung sum TF-IDF per kata
                sum_tfidf = X_full_tfidf.sum(axis=0)
                words = vectorizer.get_feature_names_out()
                
                # Buat DataFrame
                tfidf_data = [(words[i], sum_tfidf[0, i]) for i in range(len(words))]
                df_tfidf = pd.DataFrame(tfidf_data, columns=['Kata', 'Skor Total TF-IDF'])
                df_tfidf = df_tfidf.sort_values(by='Skor Total TF-IDF', ascending=False).head(10).reset_index(drop=True)
                
                c_tf1, c_tf2 = st.columns([1, 2])
                with c_tf1:
                    st.dataframe(df_tfidf)
                with c_tf2:
                    fig_tf, ax_tf = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='Skor Total TF-IDF', y='Kata', data=df_tfidf, palette='viridis', ax=ax_tf)
                    ax_tf.set_title("Top 10 Kata Paling Berbobot (TF-IDF)")
                    st.pyplot(fig_tf)
                    plt.close(fig_tf)
            except Exception as e:
                st.error(f"Gagal memuat visualisasi TF-IDF. Pastikan model sudah ditraining. Error: {e}")

            # --- PROBABILITAS PRIOR & LIKELIHOOD NAIVE BAYES ---
            st.divider()
            st.subheader("4. Probabilitas Kelas (Prior) & Likelihood - Naive Bayes")
            
            if 'model_nb_final' in st.session_state:
                # Ekstrak model dan vectorizer dari pipeline
                nb_model = st.session_state['model_nb_final']
                tfidf_vectorizer = st.session_state['vectorizer_final']
                
                # 1. Menarik nilai Prior Probability (dari model final)
                prior_probs = np.exp(nb_model.class_log_prior_)
                df_prior = pd.DataFrame({
                    "Kelas Sentimen": nb_model.classes_, 
                    "Prior Probability": prior_probs
                })
                st.markdown("**Prior Probabilities (Probabilitas Awal Kelas):**")
                st.dataframe(df_prior.style.format({"Prior Probability": "{:.6f}"}))
                
                # 2. Menarik nilai Likelihood untuk kata uji
                st.markdown("**Likelihood Kata (Probabilitas Kata Berdasarkan Kelas):**")
                input_kata_uji = st.text_input("Masukkan kata uji (pisahkan dengan koma):", value="dukung, program, makan, gizi, gratis, mantap")
                
                if input_kata_uji:
                    kata_uji = [k.strip().lower() for k in input_kata_uji.split(',') if k.strip()]
                    likelihood_data = []
                    
                    for kata in kata_uji:
                        row = {"Kata Uji": kata}
                        if kata in tfidf_vectorizer.vocabulary_:
                            idx = tfidf_vectorizer.vocabulary_[kata]
                            likelihood_asli = np.exp(nb_model.feature_log_prob_[:, idx])
                            for i, cls in enumerate(nb_model.classes_):
                                row[f"Likelihood '{cls}'"] = likelihood_asli[i]
                        else:
                            for cls in nb_model.classes_:
                                row[f"Likelihood '{cls}'"] = "Tidak ada di vocabulary"
                        likelihood_data.append(row)
                        
                    st.dataframe(pd.DataFrame(likelihood_data))

            # --- BOBOT & BIAS SVM ---
            st.divider()
            st.subheader("5. Bobot (Weights) & Bias - Support Vector Machine (SVM)")
            
            if 'model_svm_final' in st.session_state:
                svm_model = st.session_state['model_svm_final']
                tfidf_vectorizer_svm = st.session_state['vectorizer_final']
                
                st.write(f"**Model SVM:** `LinearSVC` (Kernel is always linear)")
                
                # 1. Menarik nilai Bias (Intercept)
                st.markdown("**Nilai Bias (Intercept) SVM:**")
                st.caption("*Catatan: Jika data Anda terdiri dari 2 kelas (misal Negatif & Positif), hanya ada 1 batas keputusan. Jika 3 kelas, akan ada 3 kombinasi batas keputusan.*")
                bias_df = pd.DataFrame({
                    "Fungsi Keputusan (Decision Function)": [f"Fungsi ke-{i}" for i in range(len(svm_model.intercept_))], 
                    "Nilai Bias (b)": svm_model.intercept_
                })
                st.dataframe(bias_df)
                
                # 2. Menarik nilai Bobot untuk kata uji
                st.markdown("**Bobot Kata (Koefisien w):**")
                input_kata_uji_svm = st.text_input("Masukkan kata uji SVM (pisahkan dengan koma):", value="dukung, program, makan, gizi, gratis, mantap", key="input_svm")
                
                if input_kata_uji_svm:
                    kata_uji_svm = [k.strip().lower() for k in input_kata_uji_svm.split(',') if k.strip()]
                    svm_weights_data = []
                    
                    # Pastikan coef_ berbentuk dense array dari sparse matrix agar bisa dipanggil per indeks
                    coef_array = svm_model.coef_.toarray() if hasattr(svm_model.coef_, "toarray") else np.array(svm_model.coef_)
                    
                    for kata in kata_uji_svm:
                        row = {"Kata Uji": kata}
                        if kata in tfidf_vectorizer_svm.vocabulary_:
                            idx = tfidf_vectorizer_svm.vocabulary_[kata]
                            for i in range(coef_array.shape[0]):
                                row[f"Bobot 'w' (Fungsi {i})"] = coef_array[i, idx]
                        else:
                            for i in range(coef_array.shape[0]):
                                row[f"Bobot 'w' (Fungsi {i})"] = "Tidak ada di vocabulary"
                        svm_weights_data.append(row)
                        
                    st.dataframe(pd.DataFrame(svm_weights_data))

            if 'best_model' in st.session_state:
                st.button("Lanjut ke Evaluasi 👉", on_click=set_page, args=(PAGES[4],))
        else:
            st.info("Silakan latih model baru untuk melihat hasil evaluasi.")
    else:
        st.warning("Data belum dilabeli.")

# --- TAB 5: EVALUASI PER ASPEK ---
elif menu == "5. Evaluasi Detail (Per Aspek)":
    st.header("Evaluasi Performa Model (Global & Per Aspek)")
    
    if 'cv_nb_results' in st.session_state and 'cv_svm_results' in st.session_state and 'df_model_for_eval' in st.session_state:
        # Use CV results for global evaluation
        cv_nb_results = st.session_state['cv_nb_results']
        cv_svm_results = st.session_state['cv_svm_results']
        df_model_for_eval = st.session_state['df_model_for_eval']
        
        # --- 1. EVALUASI GLOBAL ---
        st.subheader("1. Evaluasi Global (Data Testing 20%)")
        
        # Re-calculate metrics from CV results for display
        global_nb_metrics, _, _ = evaluate_model(cv_nb_results, "Naive Bayes")
        global_svm_metrics, _, _ = evaluate_model(cv_svm_results, "LinearSVC")

        global_nb = {k: v for k, v in global_nb_metrics.items() if k != 'train_time'}
        global_nb['Model'] = global_nb.pop('model')
        global_svm = {k: v for k, v in global_svm_metrics.items() if k != 'train_time'}
        global_svm['Model'] = global_svm.pop('model')
        
        global_df = pd.DataFrame([global_nb, global_svm]).set_index("Model")
        st.dataframe(global_df.style.highlight_max(axis=0, color='green').format("{:.2%}"))
        
        # Grafik Global
        fig_global, ax_global = plt.subplots(figsize=(8, 4))
        global_melted = global_df.reset_index().melt(id_vars='Model', var_name='Metrik', value_name='Skor')
        sns.barplot(data=global_melted, x='Metrik', y='Skor', hue='Model', palette='viridis', ax=ax_global)
        ax_global.set_ylim(0, 1.1)
        ax_global.set_title("Perbandingan Metrik Global")
        for p in ax_global.patches:
            ax_global.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=9)
        st.pyplot(fig_global)
        plt.close(fig_global)
        
        st.divider()

        # --- 2. EVALUASI PER ASPEK ---
        # For aspect evaluation, we need to map the CV predictions back to the original df_model
        st.subheader("2. Evaluasi Detail per Aspek")
        
        # Create a temporary DataFrame with CV predictions and original aspects
        temp_eval_df = pd.DataFrame({
            'y_true': cv_nb_results['y_true'], # True labels are the same for both models
            'pred_nb': cv_nb_results['y_pred'],
            'pred_svm': cv_svm_results['y_pred'],
            'aspect_list': df_model_for_eval['aspect_list']
        }).explode('aspect_list').dropna(subset=['aspect_list'])
        
        aspect_metrics = []
        for aspect in temp_eval_df['aspect_list'].unique():
            subset = temp_eval_df[temp_eval_df['aspect_list'] == aspect]
            if not subset.empty:
                # ... (calculate metrics for subset as before)
                aspect_metrics.append({ # Re-calculate metrics for each aspect
                    'Aspek': aspect,
                    'Jumlah Segmen': len(subset),
                    'Akurasi NB': accuracy_score(subset['y_true'], subset['pred_nb']),
                    'Akurasi SVM': accuracy_score(subset['y_true'], subset['pred_svm']),
                    'F1-Score NB': f1_score(subset['y_true'], subset['pred_nb'], average='weighted', zero_division=0),
                    'F1-Score SVM': f1_score(subset['y_true'], subset['pred_svm'], average='weighted', zero_division=0)
                })
        
        metrics_df = pd.DataFrame(aspect_metrics).sort_values(by='Jumlah Segmen', ascending=False)
        st.dataframe(metrics_df.style.format({'Akurasi NB': '{:.2%}', 'Akurasi SVM': '{:.2%}', 'F1-Score NB': '{:.2%}', 'F1-Score SVM': '{:.2%}'}))
        
        # Grafik Per Aspek
        st.subheader("Grafik Akurasi per Aspek")
        plot_df = metrics_df.melt(id_vars=['Aspek'], value_vars=['Akurasi NB', 'Akurasi SVM'], var_name='Model', value_name='Akurasi')
        plot_df['Model'] = plot_df['Model'].str.replace('Akurasi ', '')
        
        fig_aspect, ax_aspect = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_df, x='Aspek', y='Akurasi', hue='Model', palette='coolwarm', ax=ax_aspect)
        ax_aspect.set_ylim(0, 1.1)
        ax_aspect.set_title("Perbandingan Akurasi NB vs SVM per Aspek")
        plt.xticks(rotation=45)
        for p in ax_aspect.patches:
             if p.get_height() > 0:
                ax_aspect.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8)
        st.pyplot(fig_aspect)
        plt.close(fig_aspect)
        
        # --- WORDCLOUD PER ASPEK ---
        st.divider()
        st.subheader("Visualisasi WordCloud per Aspek (Berdasarkan Data Asli)")
        aspects_list = metrics_df['Aspek'].tolist()
        if aspects_list:
            tabs = st.tabs(aspects_list)
            for i, aspect in enumerate(aspects_list):
                with tabs[i]:
                    # Use the original df_model for wordcloud, filtered by aspect
                    # This needs to be done carefully to match segments to aspects
                    subset = df_model_for_eval[df_model_for_eval['aspect_list'].apply(lambda x: aspect in x if isinstance(x, list) else False)]
                    
                    wc_col1, wc_col2, wc_col3 = st.columns(3)
                    
                    # POSITIF
                    with wc_col1:
                        st.markdown("##### Sentimen Positif")
                        subset_pos = subset[subset['sentiment_label'] == 'Positif']
                        text_pos = " ".join(subset_pos['segment'].astype(str))
                        if text_pos.strip():
                            wc_pos = WordCloud(width=300, height=200, background_color='white', colormap='Greens').generate(text_pos)
                            fig_wc, ax_wc = plt.subplots()
                            ax_wc.imshow(wc_pos, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        else:
                            st.caption("Tidak ada kata positif.")

                    # NEGATIF
                    with wc_col2:
                        st.markdown("##### Sentimen Negatif")
                        subset_neg = subset[subset['sentiment_label'] == 'Negatif']
                        text_neg = " ".join(subset_neg['segment'].astype(str))
                        if text_neg.strip():
                            wc_neg = WordCloud(width=300, height=200, background_color='white', colormap='Reds').generate(text_neg)
                            fig_wc, ax_wc = plt.subplots()
                            ax_wc.imshow(wc_neg, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        else:
                            st.caption("Tidak ada kata negatif.")

                    # NETRAL
                    with wc_col3:
                        st.markdown("##### Sentimen Netral")
                        subset_neu = subset[subset['sentiment_label'] == 'Netral']
                        text_neu = " ".join(subset_neu['segment'].astype(str))
                        if text_neu.strip():
                            wc_neu = WordCloud(width=300, height=200, background_color='white', colormap='Blues').generate(text_neu)
                            fig_wc, ax_wc = plt.subplots()
                            ax_wc.imshow(wc_neu, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        else:
                            st.caption("Tidak ada kata netral.")

        st.button("Lanjut ke Prediksi 👉", on_click=set_page, args=(PAGES[5],))
    else:
        st.error("Model belum dilatih.")

# --- TAB 6: PREDIKSI MANUAL ---
elif menu == "6. Prediksi Manual":
    st.header("Uji Coba Prediksi Manual")
    # Check for the new session state keys
    if 'best_model' not in st.session_state or 'vectorizer_final' not in st.session_state:
        st.error("Model belum dilatih! Anda tidak bisa melakukan prediksi manual.")
        st.warning("Silakan kembali ke 'Tab 4. Modeling', jalankan training, lalu kembali ke sini.")
    else:
        st.info(f"Menggunakan model **{st.session_state['best_model_name']}** yang telah dilatih. Prediksi dilakukan per segmen kalimat.")
        
        best_model = st.session_state['best_model']
        vectorizer_final = st.session_state['vectorizer_final']
        
        user_input = st.text_area("Masukkan teks atau tweet yang ingin diprediksi sentimennya:", height=150)
        
        if st.button("Prediksi Sentimen"):
            if user_input.strip() == "":
                st.warning("Teks tidak boleh kosong. Masukkan kalimat untuk diuji.")
            else:
                with st.spinner("Memproses teks dan melakukan prediksi..."):
                    
                    # 1. Preprocessing
                    processed_segments = preprocess_text(user_input)
                    
                    if not processed_segments:
                        st.warning("Teks tidak menghasilkan segmen yang valid setelah dibersihkan (mungkin isinya hanya angka/simbol).")
                        st.stop()

                    # 2. TF-IDF Transformation using the final vectorizer
                    X_processed_tfidf = vectorizer_final.transform(processed_segments)

                    # 3. Prediksi Kelas menggunakan model terbaik
                    preds = best_model.predict(X_processed_tfidf)
                    
                    # 4. Prediksi Probabilitas (Persentase Keyakinan)
                    probs = None
                    try:
                        # Only try predict_proba if the model supports it (e.g., not LinearSVC by default)
                        if hasattr(best_model, 'predict_proba'):
                            probs = best_model.predict_proba(X_processed_tfidf) # This will work for NB
                    except AttributeError:
                        probs = None # LinearSVC does not have predict_proba by default

                    st.divider()
                    st.subheader("Hasil Bedah Prediksi")
                    
                    st.markdown("**Teks Asli:**")
                    st.write(f"> *{user_input}*")
                    
                    def get_color(label):
                        if str(label).lower() == 'positif': return "🟢 Positif"
                        elif str(label).lower() == 'negatif': return "🔴 Negatif"
                        return "⚪ Netral"
                    
                    results_data = []
                    classes = best_model.classes_ if hasattr(best_model, 'classes_') else ['Negatif', 'Netral', 'Positif'] # Fallback
                    
                    for i, segment in enumerate(processed_segments):
                        aspects = get_aspects(segment)
                        
                        prob_str = "Probabilitas tidak tersedia"
                        if probs is not None:
                            prob_str = " | ".join([f"{cls}: {probs[i][j]:.1%}" for j, cls in enumerate(classes)])
                        else:
                            prob_str = "Probabilitas tidak tersedia (LinearSVC)"
                            
                        results_data.append({
                            "Segmen Teks Bersih": segment,
                            "Deteksi Aspek": ", ".join(aspects),
                            "Label Prediksi": get_color(preds[i]),
                            "Keyakinan": prob_str
                        })
                    
                    st.table(pd.DataFrame(results_data))
                    
                    # --- INSIGHT DIAGNOSTIK OTOMATIS ---
                    st.markdown("### 💡 Diagnostik Kesalahan (Baca Ini Jika Prediksi Salah)")
                    st.markdown("""
                    1. **Lihat 'Segmen Teks Bersih':** Apakah ada kata penting (seperti kata 'tidak') yang terhapus oleh Stopword? Jika iya, pantas saja sentimennya berbalik.
                2. **Lihat 'Deteksi Aspek':** Aspek diekstrak berdasarkan kata kunci yang tersisa setelah dibersihkan. Jika teks bersihnya adalah `"makan buruk"`, tapi tidak ada kata `"buruk"` di daftar kamus Aspek Anda (di baris awal kode), ia akan dilempar ke `"Lainnya"`.
                    3. **Lihat 'Keyakinan (Probabilitas)':** Jika persentase Positif `49%` dan Negatif `51%`, artinya model Anda sebenarnya ragu-ragu karena data latihnya kurang variatif.
                    """)