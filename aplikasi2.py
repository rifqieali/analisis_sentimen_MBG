"""
Aplikasi ABSA (Aspect-Based Sentiment Analysis) - Program MBG
Logika: absaMBG.py (K-Fold CV, Pipeline Lanjutan, TF-IDF Opt)
UI/Tampilan: absa_mbg.py (Elegan, CSS Custom, Multi-page)
"""

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
import warnings
from io import BytesIO

from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURASI HALAMAN & CSS ELEGANT
# ==========================================
st.set_page_config(page_title="Analisis Sentimen MBG", layout="wide", page_icon="📑")

st.markdown("""
    <style>
    .main {background-color: #FAFAFA;}
    h1, h2, h3 {color: #2C3E50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stButton>button {border-radius: 5px; border: 1px solid #BDC3C7;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. FUNGSI CACHED & RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    resources = ['punkt']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    
    factory_stop = StopWordRemoverFactory()
    
    sw_custom = {
        "yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp",
        "biar", "bikin", "udah", "udh", "aja", "sih", "deh", "nih",
        "lah", "dong", "kan", "tuh", "mah", "wkwk", "haha", "hehe",
    }
    negation_words = {'tidak', 'tak', 'tiada', 'bukan', 'jangan', 'belum', 'kurang', 'gak', 'ga', 'nggak', 'enggak'}

    stopwords_sastrawi = set(factory_stop.get_stop_words())
    final_stopwords = (stopwords_sastrawi | sw_custom) - negation_words
    
    return final_stopwords, negation_words

final_stopwords, negation_words = load_resources()

@st.cache_data
def load_normalization_dict():
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    try:
        response = requests.get(url, timeout=10)
        file_excel = BytesIO(response.content)
        df_norm = pd.read_excel(file_excel)
        norm_dict = dict(zip(df_norm.iloc[:, 0].astype(str).str.lower(), df_norm.iloc[:, 1].astype(str).str.lower()))
        return norm_dict
    except Exception:
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

# ==========================================
# 2. MACHINE LEARNING LOGIC (K-Fold CV)
# ==========================================
TFIDF_PARAMS = {
    "max_features": 3000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.9,
    "sublinear_tf": True,
}

def extract_tfidf(corpus, fit=True, vectorizer=None):
    if fit:
        tfidf = TfidfVectorizer(**TFIDF_PARAMS)
        X = tfidf.fit_transform(corpus)
        return X, tfidf
    else:
        if vectorizer is None: raise ValueError("Vectorizer must be provided if fit is False.")
        X = vectorizer.transform(corpus)
        return X, vectorizer

def train_model_nb(X, y):
    model = MultinomialNB(alpha=1.0)
    model.fit(X, y)
    return model

def train_model_svm_linear(X, y):
    model = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    model.fit(X, y)
    return model

def cross_validation_pipeline(corpus, labels, model_type, progress_bar=None, n_splits=10):
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_all = np.empty_like(labels.values, dtype=object)
    
    for fold, (train_index, test_index) in enumerate(skf.split(corpus, y_encoded)):
        X_train, X_test = corpus.iloc[train_index], corpus.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
        X_train_tfidf, tfidf_vectorizer = extract_tfidf(X_train, fit=True)
        X_test_tfidf, _ = extract_tfidf(X_test, fit=False, vectorizer=tfidf_vectorizer)
        
        if model_type == "nb": model = train_model_nb(X_train_tfidf, y_train)
        elif model_type == "svm": model = train_model_svm_linear(X_train_tfidf, y_train)
            
        y_pred = model.predict(X_test_tfidf)
        y_pred_all[test_index] = y_pred
        if progress_bar: progress_bar.progress((fold + 1) / n_splits)
            
    return {"y_true": labels.tolist(), "y_pred": y_pred_all.tolist(), "model_name": "Naive Bayes" if model_type == "nb" else "LinearSVC"}

def evaluate_model(cv_results, model_name):
    y_true = cv_results["y_true"]
    y_pred = cv_results["y_pred"]
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    cr = classification_report(y_true, y_pred, zero_division=0)
    labels = sorted(np.unique(np.concatenate((y_true, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title(f"Confusion Matrix {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return metrics, fig, cr

# ==========================================
# 3. FUNGSI PREPROCESSING & SEGMENTASI
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    words = text.split()
    words = [norm_dict_global.get(w, w) for w in words]
    return ' '.join(words)

def segmentasi_kalimat(text):
    konjungsi_list = {'tetapi', 'namun', 'dan', 'karena', 'meskipun', 'tapi', 'sedangkan'}
    konjungsi_regex = r'\b(' + '|'.join(konjungsi_list) + r')\b'
    segmen = [s.strip() for s in re.split(konjungsi_regex, text) if s and s.strip() and s.strip().lower() not in konjungsi_list]
    return segmen if segmen else [text]

def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in final_stopwords]
    return ' '.join(words)

def preprocess_text(text):
    text_clean = clean_text(text)
    text_norm = normalize_text(text_clean)
    segmen_list = segmentasi_kalimat(text_norm)
    final_segments = []
    for seg in segmen_list:
        processed = remove_stopwords(seg)
        if processed:
            final_segments.append(processed)
    return final_segments

def get_aspect(text):
    text_words = set(str(text).split())
    keywords = {
        'Kualitas': ["kualitas", "bagus", "jelek", "enak", "basi", "gizi", "susu", "menu", "rasa", "porsi", "higienis", "keracunan", "sehat", "mentah", "keras", "hambar", "ulat", "lauk", "sayur", "karbohidrat", "protein", "lemak", "gula", "ayam", "telur", "kenyang", "alergi", "higienitas"],
        'Layanan': ["layan", "antri", "ramah", "lambat", "cepat", "bantu", "saji", "distribusi", "vendor", "katering", "sekolah", "siswa", "guru", "telat", "molor", "bocor", "tepat waktu", "pelosok", "merata", "zonasi", "umkm", "kemasan", "kotak", "plastik"],
        'Anggaran': ["harga", "mahal", "murah", "biaya", "bayar", "anggar", "boros", "korupsi", "dana", "apbn", "pajak", "potong", "sunat", "markup", "tender", "proyek", "apbd", "defisit", "utang", "ekonomi", "alokasi", "transparan"]
    }
    for aspect, keys in keywords.items():
        if not text_words.isdisjoint(keys):
            return aspect
    return 'Lainnya'

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
# 4. INTERFACE STREAMLIT UTAMA
# ==========================================
st.title("📊 Analisis Sentimen Program Makan Bergizi Gratis")
st.markdown("Aplikasi berbasis InSet Lexicon & Machine Learning dengan **Segmentasi Konjungsi**")

PAGES = [
    "1. Upload Data", 
    "2. Preprocessing & Segmentasi", 
    "3. Labeling & Aspek", 
    "4. Modeling (Training)", 
    "5. Evaluasi Detail (Per Aspek)",
    "6. Klasifikasi Manual"
]

if 'current_page' not in st.session_state: st.session_state['current_page'] = PAGES[0]
if 'df_raw' not in st.session_state: st.session_state['df_raw'] = None
if 'df_processed' not in st.session_state: st.session_state['df_processed'] = None

def set_page(page_name):
    st.session_state['current_page'] = page_name

st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.selectbox("Pilih Tahapan", PAGES, key='current_page')

# --- TAB 1: UPLOAD DATA ---
if menu == PAGES[0]:
    st.header("1. Upload Dataset CSV")
    st.write("Sistem menerima data mentah atau data yang telah dipreprocessing (melewati tahap 2).")
    
    data_type = st.radio("Jenis Data:", ["Data Mentah (Raw)", "Data Preprocessed (Terdapat kolom 'segment')"])
    uploaded_file = st.file_uploader("Format yang didukung: CSV", type="csv")
    
    if uploaded_file:
        # PROTOKOL HARD RESET (SAPU JAGAT)
        keys_to_destroy = [
            'df_processed', 'model_nb', 'model_svm', 'eval_data', 'cv_nb_results', 
            'cv_svm_results', 'model_nb_final', 'model_svm_final', 'vectorizer_final', 'df_model_for_eval'
        ]
        for key in keys_to_destroy:
            if key in st.session_state:
                del st.session_state[key]
                
        df = pd.read_csv(uploaded_file)
        
        if data_type == "Data Mentah (Raw)":
            st.session_state['df_raw'] = df
            st.success(f"✅ Data mentah baru berhasil dimuat! Jumlah dokumen: {len(df)}")
            st.info("Sistem telah dibersihkan dari memori lama. Anda siap memproses data baru dari nol.")
            st.dataframe(df.head())
            st.button("Lanjut ke Preprocessing 👉", on_click=set_page, args=(PAGES[1],))
        else:
            if 'segment' not in df.columns:
                st.error("❌ File CSV tidak valid! Pastikan terdapat kolom 'segment'.")
            else:
                st.session_state['df_raw'] = None
                st.session_state['df_processed'] = df
                st.success(f"✅ Data hasil preprocessing dimuat! Jumlah segmen: {len(df)}")
                st.info("Sistem telah dibersihkan dari memori lama. Anda siap melabeli data baru.")
                st.dataframe(df.head())
                st.button("Lanjut ke Labeling 👉", on_click=set_page, args=(PAGES[2],))

# --- TAB 2: PREPROCESSING ---
elif menu == PAGES[1]:
    st.header("2. Preprocessing & Segmentasi")
    st.warning("PERHATIAN: 1 baris teks dapat pecah menjadi beberapa baris segmen berdasarkan konjungsi (dan, tetapi, namun).")
    
    if st.session_state.get('df_processed') is not None:
        st.info("Data preprocessed sudah tersedia di memori. Anda dapat melanjutkan ke tab Labeling.")
        st.dataframe(st.session_state['df_processed'].head())
        st.button("Lanjut ke Labeling 👉", on_click=set_page, args=(PAGES[2],))
        
    elif st.session_state.get('df_raw') is not None:
        df_raw = st.session_state['df_raw'].copy()
        col_text = st.selectbox("Pilih kolom teks:", df_raw.columns)
        
        if st.button("⚙️ Mulai Preprocessing / Jalankan Ulang"):
            with st.spinner("Membersihkan teks, normalisasi, menghapus stopword, dan memecah segmen..."):
                my_bar = st.progress(0)
                records = []
                total_rows = len(df_raw)
                
                for idx, row in df_raw.iterrows():
                    segments = preprocess_text(str(row[col_text]))
                    for seg in segments:
                        records.append({"doc_id": idx, "segment": seg})
                    
                    if idx % max(1, total_rows // 100) == 0:
                        my_bar.progress(min(idx / total_rows, 1.0))
                my_bar.progress(1.0)
                
                df_out = pd.DataFrame(records)
                st.session_state['df_processed'] = df_out
                st.success(f"✅ Pemrosesan selesai. {total_rows} dokumen dipecah menjadi {len(df_out)} segmen.")
                
        if st.session_state.get('df_processed') is not None:
            st.dataframe(st.session_state['df_processed'][['doc_id', 'segment']].head(10))
            csv = st.session_state['df_processed'].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Preprocessing", csv, "preprocessed_data.csv", "text/csv")
            st.write("")
            st.button("Lanjut ke Labeling 👉", on_click=set_page, args=(PAGES[2],))
    else:
        st.warning("Silakan upload data di Tab 1 terlebih dahulu.")

# --- TAB 3: LABELING ---
elif menu == PAGES[2]:
    st.header("3. Labeling Aspek & Sentimen (Level Segmen)")
    
    if st.session_state.get('df_processed') is not None:
        df = st.session_state['df_processed']
        
        if st.button("Jalankan Pelabelan & Aspek"):
            with st.spinner("Klasifikasi InSet Lexicon dan Rule-based Aspect..."):
                df['aspek'] = df['segment'].apply(get_aspect)
                df['sentimen'] = df['segment'].apply(determine_sentiment)
                st.session_state['df_processed'] = df
                st.success("✅ Pelabelan selesai.")
                
        if 'sentimen' in df.columns:
            c1, c2 = st.columns(2)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Distribusi Sentimen")
                fig, ax = plt.subplots()
                labels = df['sentimen'].value_counts().index
                sizes = df['sentimen'].value_counts().values
                ax.pie(sizes, labels=labels, autopct='%1.1f%%')
                st.pyplot(fig)
            with c2:
                st.subheader("Distribusi Aspek")
                st.bar_chart(df['sentimen'].value_counts())
            with c3:
                st.subheader("Distribusi Sentimen")
                st.bar_chart(df['aspek'].value_counts())
            
            st.dataframe(df[['segment', 'aspek', 'sentimen']].head(10))
            st.divider()
            
            # Wordcloud dari file awal
            st.subheader("Visualisasi WordCloud")
            wc_c1, wc_c2, wc_c3 = st.columns(3)
            with wc_c1:
                st.markdown("**Sentimen Positif**")
                text_pos = " ".join(df[df['sentimen'] == 'Positif']['segment'].astype(str))
                if text_pos.strip():
                    wc_pos = WordCloud(width=300, height=200, background_color='white', colormap='Greens').generate(text_pos)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_pos, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
            with wc_c2:
                st.markdown("**Sentimen Negatif**")
                text_neg = " ".join(df[df['sentimen'] == 'Negatif']['segment'].astype(str))
                if text_neg.strip():
                    wc_neg = WordCloud(width=300, height=200, background_color='white', colormap='Reds').generate(text_neg)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_neg, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
            with wc_c3:
                st.markdown("**Sentimen Netral**")
                text_neu = " ".join(df[df['sentimen'] == 'Netral']['segment'].astype(str))
                if text_neu.strip():
                    wc_neu = WordCloud(width=300, height=200, background_color='white', colormap='Blues').generate(text_neu)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_neu, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)

            st.button("Lanjut ke Modeling 👉", on_click=set_page, args=(PAGES[3],))
    else:
        st.warning("Selesaikan tahap Preprocessing terlebih dahulu.")

# --- TAB 4: MODELING (K-FOLD) ---
elif menu == PAGES[3]:
    st.header("4. Pelatihan Model Machine Learning (K-Fold CV)")
    
    if st.session_state.get('df_processed') is not None and 'sentimen' in st.session_state['df_processed'].columns:
        df = st.session_state['df_processed']
        remove_neutral = st.checkbox("Hapus Data Netral?", value=True)
        df_model = df[df['sentimen'] != 'Netral'] if remove_neutral else df.copy()
            
        if st.button("Mulai Training Model"):
            with st.spinner("Melatih model dengan optimasi 10-Fold CV... (Mohon tunggu)"):
                X = df_model['segment']
                y = df_model['sentimen']
                
                col_eval1, col_eval2 = st.columns(2)
                
                # --- CV NAIVE BAYES ---
                with col_eval1:
                    st.write("**Melatih Naive Bayes (10-Fold CV)...**")
                    pb_nb = st.progress(0)
                    t_nb = time.perf_counter()
                    cv_nb_results = cross_validation_pipeline(X, y, model_type="nb", progress_bar=pb_nb)
                    t_nb = time.perf_counter() - t_nb
                    metrics_nb, fig_nb, cr_nb = evaluate_model(cv_nb_results, "Naive Bayes")
                    metrics_nb["train_time (s)"] = round(t_nb, 3)
                    
                    st.dataframe(pd.DataFrame([metrics_nb]).set_index("model").style.format("{:.4f}"))
                    st.pyplot(fig_nb)
                    plt.close(fig_nb)
                    with st.expander("Detail Classification Report (NB)"):
                        st.code(cr_nb)
                
                # --- CV LINEAR SVC ---
                with col_eval2:
                    st.write("**Melatih LinearSVC (10-Fold CV)...**")
                    pb_svm = st.progress(0)
                    t_svm = time.perf_counter()
                    cv_svm_results = cross_validation_pipeline(X, y, model_type="svm", progress_bar=pb_svm)
                    t_svm = time.perf_counter() - t_svm
                    metrics_svm, fig_svm, cr_svm = evaluate_model(cv_svm_results, "LinearSVC")
                    metrics_svm["train_time (s)"] = round(t_svm, 3)
                    
                    st.dataframe(pd.DataFrame([metrics_svm]).set_index("model").style.format("{:.4f}"))
                    st.pyplot(fig_svm)
                    plt.close(fig_svm)
                    with st.expander("Detail Classification Report (SVM)"):
                        st.code(cr_svm)
                        
                # --- Latih Model Final ---
                with st.spinner("Melatih model final pada seluruh data untuk ekstraksi..."):
                    X_all_tfidf, vectorizer_final = extract_tfidf(X, fit=True)
                    model_nb_final = train_model_nb(X_all_tfidf, y)
                    model_svm_final = train_model_svm_linear(X_all_tfidf, y)
                    
                    st.session_state['model_nb_final'] = model_nb_final
                    st.session_state['model_svm_final'] = model_svm_final
                    st.session_state['vectorizer_final'] = vectorizer_final
                    
                    st.session_state['cv_nb_results'] = cv_nb_results
                    st.session_state['cv_svm_results'] = cv_svm_results
                    st.session_state['df_model_for_eval'] = df_model 
                    
                st.success(f"✅ Training Selesai! Model Naive Bayes dan LinearSVC berhasil dilatih.")
        
        st.divider()
        
        if 'model_nb_final' in st.session_state and 'model_svm_final' in st.session_state:
            # --- TF-IDF VISUALIZATION ---
            st.subheader("Visualisasi Bobot TF-IDF Tertinggi")
            try:
                vectorizer = st.session_state['vectorizer_final']
                X_full_tfidf, _ = extract_tfidf(df_model['segment'], fit=False, vectorizer=vectorizer)
                sum_tfidf = X_full_tfidf.sum(axis=0)
                words = vectorizer.get_feature_names_out()
                
                df_tfidf = pd.DataFrame([(words[i], sum_tfidf[0, i]) for i in range(len(words))], columns=['Kata', 'Skor TF-IDF'])
                df_tfidf = df_tfidf.sort_values(by='Skor TF-IDF', ascending=False).head(10).reset_index(drop=True)
                
                c_tf1, c_tf2 = st.columns([1, 2])
                with c_tf1: st.dataframe(df_tfidf)
                with c_tf2:
                    fig_tf, ax_tf = plt.subplots(figsize=(6, 4))
                    sns.barplot(x='Skor TF-IDF', y='Kata', data=df_tfidf, palette='viridis', ax=ax_tf)
                    ax_tf.set_title("Top 10 Kata TF-IDF")
                    st.pyplot(fig_tf)
                    plt.close(fig_tf)
            except Exception as e:
                st.error(f"Gagal memuat visualisasi TF-IDF. Error: {e}")

            # --- PROBABILITAS NAIVE BAYES ---
            st.divider()
            st.subheader("Probabilitas Kelas (Prior) & Likelihood - Naive Bayes")
            nb_model = st.session_state['model_nb_final']
            tfidf_vectorizer = st.session_state['vectorizer_final']
            
            prior_probs = np.exp(nb_model.class_log_prior_)
            df_prior = pd.DataFrame({"Kelas Sentimen": nb_model.classes_, "Prior Probability": prior_probs})
            st.dataframe(df_prior.style.format({"Prior Probability": "{:.6f}"}))
            
            input_kata_uji = st.text_input("Masukkan kata uji Likelihood (pisahkan dengan koma):", value="dukung, program, mantap")
            if input_kata_uji:
                kata_uji = [k.strip().lower() for k in input_kata_uji.split(',') if k.strip()]
                likelihood_data = []
                for kata in kata_uji:
                    row = {"Kata Uji": kata}
                    if kata in tfidf_vectorizer.vocabulary_:
                        idx = tfidf_vectorizer.vocabulary_[kata]
                        likelihood_asli = np.exp(nb_model.feature_log_prob_[:, idx])
                        for i, cls in enumerate(nb_model.classes_): row[f"Likelihood '{cls}'"] = likelihood_asli[i]
                    else:
                        for cls in nb_model.classes_: row[f"Likelihood '{cls}'"] = "Tidak ada di vocabulary"
                    likelihood_data.append(row)
                st.dataframe(pd.DataFrame(likelihood_data))

            # --- BOBOT SVM ---
            st.divider()
            st.subheader("Bobot Kata (Weights) & Bias - LinearSVC")
            svm_model = st.session_state['model_svm_final']
            tfidf_vectorizer_svm = st.session_state['vectorizer_final']
            
            bias_df = pd.DataFrame({"Fungsi Keputusan": [f"Fungsi ke-{i}" for i in range(len(svm_model.intercept_))], "Nilai Bias (b)": svm_model.intercept_})
            st.dataframe(bias_df)
            
            input_kata_uji_svm = st.text_input("Masukkan kata uji Bobot SVM (pisahkan koma):", value="dukung, program, mantap", key="svm_in")
            if input_kata_uji_svm:
                kata_uji_svm = [k.strip().lower() for k in input_kata_uji_svm.split(',') if k.strip()]
                svm_weights_data = []
                coef_array = svm_model.coef_.toarray() if hasattr(svm_model.coef_, "toarray") else np.array(svm_model.coef_)
                
                for kata in kata_uji_svm:
                    row = {"Kata Uji": kata}
                    if kata in tfidf_vectorizer_svm.vocabulary_:
                        idx = tfidf_vectorizer_svm.vocabulary_[kata]
                        for i in range(coef_array.shape[0]): row[f"Bobot 'w' (Fungsi {i})"] = coef_array[i, idx]
                    else:
                        for i in range(coef_array.shape[0]): row[f"Bobot 'w' (Fungsi {i})"] = "Tidak ada di vocabulary"
                    svm_weights_data.append(row)
                st.dataframe(pd.DataFrame(svm_weights_data))

            st.button("Lanjut ke Evaluasi Per Aspek 👉", on_click=set_page, args=(PAGES[4],))
    else:
        st.warning("Lakukan Pelabelan di Tab 3 terlebih dahulu.")

# --- TAB 5: EVALUASI PER ASPEK ---
elif menu == PAGES[4]:
    st.header("5. Evaluasi Detail (Performa Per Aspek)")
    
    if 'cv_nb_results' in st.session_state and 'df_model_for_eval' in st.session_state:
        cv_nb_results = st.session_state['cv_nb_results']
        cv_svm_results = st.session_state['cv_svm_results']
        df_model_for_eval = st.session_state['df_model_for_eval']
        
        temp_eval_df = pd.DataFrame({
            'y_true': cv_nb_results['y_true'],
            'pred_nb': cv_nb_results['y_pred'],
            'pred_svm': cv_svm_results['y_pred'],
            'aspek': df_model_for_eval['aspek']
        })
        
        aspect_metrics = []
        for aspect in temp_eval_df['aspek'].unique():
            subset = temp_eval_df[temp_eval_df['aspek'] == aspect]
            if not subset.empty:
                aspect_metrics.append({
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
        
        fig_aspect, ax_aspect = plt.subplots(figsize=(8, 4))
        sns.barplot(data=plot_df, x='Aspek', y='Akurasi', hue='Model', palette='coolwarm', ax=ax_aspect)
        ax_aspect.set_ylim(0, 1.1)
        plt.xticks(rotation=0)
        for p in ax_aspect.patches:
             if p.get_height() > 0: ax_aspect.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8)
        st.pyplot(fig_aspect)
        plt.close(fig_aspect)

        st.button("Lanjut ke Klasifikasi Manual 👉", on_click=set_page, args=(PAGES[5],))
    else:
        st.error("Model belum dilatih di Tab 4.")

# --- TAB 6: KLASIFIKASI MANUAL ---
elif menu == PAGES[5]:
    st.header("6. Klasifikasi Manual")
    if 'model_nb_final' not in st.session_state or 'model_svm_final' not in st.session_state or 'vectorizer_final' not in st.session_state:
        st.warning("Silakan kembali ke 'Tab 4. Modeling', jalankan training terlebih dahulu.")
    else:
        st.info("Menggunakan model **Naive Bayes** dan **LinearSVC** yang telah dilatih.")
        nb_model = st.session_state['model_nb_final']
        svm_model = st.session_state['model_svm_final']
        vectorizer_final = st.session_state['vectorizer_final']
        
        user_input = st.text_area("Masukkan teks ulasan terkait Program MBG:", height=150)
        
        if st.button("Analisis Teks"):
            if user_input.strip() == "":
                st.warning("Input tidak boleh kosong.")
            else:
                with st.spinner("Memproses prediksi..."):
                    processed_segments = preprocess_text(user_input)
                    if not processed_segments:
                        st.error("Teks tidak menghasilkan segmen bermakna setelah dibersihkan.")
                        st.stop()

                    X_processed_tfidf = vectorizer_final.transform(processed_segments)
                    preds_nb = nb_model.predict(X_processed_tfidf)
                    preds_svm = svm_model.predict(X_processed_tfidf)
                    
                    probs = None
                    try:
                        if hasattr(nb_model, 'predict_proba'): probs = nb_model.predict_proba(X_processed_tfidf)
                    except AttributeError: pass

                    st.divider()
                    st.markdown("**Hasil Bedah Prediksi:**")
                    
                    def get_color(label):
                        return "🟢 Positif" if str(label) == 'Positif' else "🔴 Negatif" if str(label) == 'Negatif' else "⚪ Netral"
                    
                    results_data = []
                    classes = nb_model.classes_ if hasattr(nb_model, 'classes_') else ['Negatif', 'Netral', 'Positif']
                    
                    for i, segment in enumerate(processed_segments):
                        aspect = get_aspect(segment)
                        prob_str = "Tidak tersedia"
                        if probs is not None:
                            prob_str = " | ".join([f"{cls}: {probs[i][j]:.1%}" for j, cls in enumerate(classes)])
                            
                        results_data.append({
                            "Segmen Teks Bersih": segment,
                            "Deteksi Aspek": aspect,
                            "Prediksi NB": get_color(preds_nb[i]),
                            "Keyakinan": prob_str,
                            "Prediksi SVM": get_color(preds_svm[i])
                        })
                    
                    st.table(pd.DataFrame(results_data))
                    
                    st.markdown("### 💡 Diagnostik Kesalahan")
                    st.markdown("""
                    1. **Lihat 'Segmen Teks Bersih':** Apakah ada kata penting yang terhapus stopword?
                    2. **Lihat 'Deteksi Aspek':** Jika kata kunci aspek tidak ada dalam teks bersih, aspek akan masuk "Lainnya".
                    """)