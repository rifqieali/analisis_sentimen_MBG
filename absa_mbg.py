"""
Aplikasi ABSA (Aspect-Based Sentiment Analysis) - Program MBG
UI Berbasis Streamlit (Elegan & Profesional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import warnings

import nltk
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Analisis Sentimen MBG", layout="wide", page_icon="📑")

# Kustomisasi CSS untuk tampilan elegan
st.markdown("""
    <style>
    .main {background-color: #FAFAFA;}
    h1, h2, h3 {color: #2C3E50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stButton>button {border-radius: 5px; border: 1px solid #BDC3C7;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# RESOURCE & CACHING
# ==========================================
@st.cache_resource
def load_nltk_sastrawi():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    # Stopword Sastrawi
    factory = StopWordRemoverFactory()
    sw_sastrawi = set(factory.get_stop_words())
    
    # Stopword NLTK
    sw_nltk = set(stopwords.words("indonesian"))
    
    # Stopword Custom
    sw_custom = {
        "yg", "dg", "rt", "dgn", "ny", "d", "klo", "kalo", "amp",
        "biar", "bikin", "udah", "udh", "aja", "sih", "deh", "nih",
        "lah", "dong", "kan", "tuh", "mah", "wkwk", "haha", "hehe",
    }
    negation_words = {'tidak', 'tak', 'tiada', 'bukan', 'jangan', 'belum', 'kurang', 'gak', 'ga', 'nggak', 'enggak'}

    # Gabungkan semua dan kurangi kata negasi
    final_stopwords = (sw_sastrawi | sw_nltk | sw_custom) - negation_words
    return final_stopwords, negation_words

final_stopwords, negation_words = load_nltk_sastrawi()

@st.cache_data
def load_kamus_normalisasi():
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        xls = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
        col_names = xls.columns.tolist()
        return dict(zip(xls[col_names[0]].astype(str).str.lower(), xls[col_names[1]].astype(str).str.lower()))
    except Exception:
        return {"yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "bgt": "banget"}

norm_dict = load_kamus_normalisasi()

@st.cache_data
def load_inset():
    pos_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    neg_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    try:
        pos_df = pd.read_csv(pos_url, sep="\t", header=None, names=["word", "weight"])
        neg_df = pd.read_csv(neg_url, sep="\t", header=None, names=["word", "weight"])
        pos_dict = dict(zip(pos_df["word"].str.lower(), pd.to_numeric(pos_df["weight"], errors='coerce').fillna(0)))
        neg_dict = dict(zip(neg_df["word"].str.lower(), pd.to_numeric(neg_df["weight"], errors='coerce').fillna(0)))
        return pos_dict, neg_dict
    except Exception:
        return {}, {}

pos_lexicon, neg_lexicon = load_inset()

# Dictionary Aspek
ASPEK_DICT = {
    "Kualitas": ["kualitas", "bagus", "jelek", "enak", "basi", "gizi", "susu", "menu", "rasa", "porsi", "higienis", "keracunan", "sehat", "mentah", "keras", "hambar", "ulat", "lauk", "sayur", "karbohidrat", "protein", "lemak", "gula", "ayam", "telur", "kenyang", "alergi", "higienitas"],
    "Anggaran": ["harga", "mahal", "murah", "biaya", "bayar", "anggar", "boros", "korupsi", "dana", "apbn", "pajak", "potong", "sunat", "markup", "tender", "proyek", "apbd", "defisit", "utang", "ekonomi", "alokasi", "transparan"],
    "Layanan": ["layan", "antri", "ramah", "lambat", "cepat", "bantu", "saji", "distribusi", "vendor", "katering", "sekolah", "siswa", "guru", "telat", "molor", "bocor", "tepat waktu", "pelosok", "merata", "zonasi", "umkm", "kemasan", "kotak", "plastik"]
}
KONJUNGSI_SEGMEN = ["tetapi", "namun", "dan", "karena", "meskipun"]

# ==========================================
# FUNGSI PEMROSESAN
# ==========================================
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", str(text))
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()

def split_segments(text: str) -> list:
    pattern = r"\b(" + "|".join(KONJUNGSI_SEGMEN) + r")\b"
    parts = re.split(pattern, text)
    segments = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        if part in KONJUNGSI_SEGMEN and i + 1 < len(parts):
            segments.append((part + " " + parts[i + 1]).strip())
            i += 2
        else:
            segments.append(part)
            i += 1
    return [s for s in segments if len(s.split()) >= 3]

def process_pipeline(text: str) -> list:
    cleaned = clean_text(text)
    tokens = [norm_dict.get(t, t) for t in cleaned.split()]
    norm_text = " ".join(tokens)
    
    segments = split_segments(norm_text)
    final_segments = []
    for seg in segments:
        words = [w for w in seg.split() if w not in final_stopwords]
        if words:
            final_segments.append(" ".join(words))
    return final_segments

def get_aspect(text: str):
    tokens = set(text.split())
    for aspek, keywords in ASPEK_DICT.items():
        if not tokens.isdisjoint(keywords):
            return aspek
    return "Lainnya"

def get_sentiment(text: str):
    tokens = text.split()
    pos_score = sum(pos_lexicon.get(t, 0) for t in tokens)
    neg_score = sum(abs(neg_lexicon.get(t, 0)) for t in tokens)
    
    if pos_score > neg_score: return "Positif"
    elif neg_score > pos_score: return "Negatif"
    return "Netral"

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
PAGES = [
    "1. Upload Data", 
    "2. Preprocessing & Segmentasi", 
    "3. Labeling & Aspek", 
    "4. Modeling (Training)", 
    "5. Evaluasi Detail (Per Aspek)",
    "6. Klasifikasi Manual"
]

if 'df' not in st.session_state: st.session_state['df'] = None
if 'df_processed' not in st.session_state: st.session_state['df_processed'] = None

# ==========================================
# UI UTAMA
# ==========================================
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio("Pilih Halaman:", PAGES)

# --- 1. UPLOAD DATA ---
if menu == PAGES[0]:
    st.title("1. Upload Dataset")
    st.write("Sistem menerima data mentah atau data yang telah dipreprocessing (melewati tahap 2).")
    
    data_type = st.radio("Jenis Data:", ["Data Mentah (Raw)", "Data Preprocessed (Terdapat kolom 'segment')"])
    uploaded_file = st.file_uploader("Format yang didukung: CSV", type="csv")
    
    if uploaded_file:
        # PROTOKOL HARD RESET: Bersihkan semua memori analisis sebelumnya
        for key in ['df_processed', 'model_nb', 'model_svm', 'eval_data']:
            if key in st.session_state:
                del st.session_state[key]
                
        df = pd.read_csv(uploaded_file)
        
        if data_type == "Data Mentah (Raw)":
            st.session_state['df'] = df
            st.success(f"Data mentah dimuat. Total dokumen: {len(df)}")
            st.dataframe(df.head())
        else:
            if 'segment' not in df.columns:
                st.error("Kolom 'segment' tidak ditemukan. Pastikan data berasal dari hasil preprocessing sistem ini.")
            else:
                st.session_state['df'] = None
                st.session_state['df_processed'] = df
                st.success(f"Data preprocessed dimuat. Total segmen: {len(df)}")
                st.dataframe(df.head())

# --- 2. PREPROCESSING ---
elif menu == PAGES[1]:
    st.title("2. Preprocessing & Segmentasi")
    
    if st.session_state.get('df_processed') is not None:
        st.info("Data preprocessed sudah tersedia di memori. Anda dapat melanjutkan ke tab Labeling.")
        st.dataframe(st.session_state['df_processed'].head())
        
    elif st.session_state.get('df') is not None:
        df_raw = st.session_state['df']
        col_text = st.selectbox("Pilih kolom teks:", df_raw.columns)
        
        if st.button("Mulai Preprocessing"):
            with st.spinner("Membersihkan teks, normalisasi, menghapus stopword, dan memecah segmen..."):
                records = []
                for idx, row in df_raw.iterrows():
                    segments = process_pipeline(str(row[col_text]))
                    for seg in segments:
                        records.append({"doc_id": idx, "segment": seg})
                
                df_out = pd.DataFrame(records)
                st.session_state['df_processed'] = df_out
                st.success(f"Pemrosesan selesai. {len(df_raw)} dokumen dipecah menjadi {len(df_out)} segmen.")
                
        if st.session_state['df_processed'] is not None:
            st.dataframe(st.session_state['df_processed'].head())
            csv = st.session_state['df_processed'].to_csv(index=False).encode('utf-8')
            st.download_button("Unduh Hasil Preprocessing", csv, "preprocessed_data.csv", "text/csv")
    else:
        st.warning("Silakan upload data di Tab 1.")

# --- 3. LABELING ---
elif menu == PAGES[2]:
    st.title("3. Labeling Aspek & Sentimen")
    
    if st.session_state.get('df_processed') is not None:
        df = st.session_state['df_processed']
        
        if st.button("Mulai Pelabelan"):
            with st.spinner("Klasifikasi InSet Lexicon dan Rule-based Aspect..."):
                df['aspek'] = df['segment'].apply(get_aspect)
                df['sentimen'] = df['segment'].apply(get_sentiment)
                st.session_state['df_processed'] = df
                st.success("Pelabelan selesai.")
                
        if 'sentimen' in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribusi Sentimen")
                st.bar_chart(df['sentimen'].value_counts())
            with c2:
                st.subheader("Distribusi Aspek")
                st.bar_chart(df['aspek'].value_counts())
            st.dataframe(df[['segment', 'aspek', 'sentimen']].head(10))
    else:
        st.warning("Selesaikan tahap Preprocessing terlebih dahulu.")

# --- 4. MODELING ---
elif menu == PAGES[3]:
    st.title("4. Pelatihan Model (Training)")
    
    if st.session_state.get('df_processed') is not None and 'sentimen' in st.session_state['df_processed'].columns:
        df = st.session_state['df_processed']
        
        rm_neutral = st.checkbox("Hapus kelas Netral?", value=True)
        df_model = df[df['sentimen'] != 'Netral'].copy() if rm_neutral else df.copy()
        
        if st.button("Latih Model (NB & SVM)"):
            with st.spinner("Ekstraksi fitur TF-IDF dan pelatihan model..."):
                X = df_model['segment']
                y = df_model['sentimen']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), sublinear_tf=True)
                X_train_vec = tfidf.fit_transform(X_train)
                X_test_vec = tfidf.transform(X_test)
                
                # Naive Bayes
                nb = MultinomialNB()
                nb.fit(X_train_vec, y_train)
                y_pred_nb = nb.predict(X_test_vec)
                
                # Linear SVC (Cepat & Stabil untuk teks)
                svm = LinearSVC(class_weight='balanced', random_state=42)
                svm.fit(X_train_vec, y_train)
                y_pred_svm = svm.predict(X_test_vec)
                
                # Simpan ke Session
                st.session_state['model_nb'] = nb
                st.session_state['model_svm'] = svm
                st.session_state['vectorizer'] = tfidf
                
                eval_data = pd.DataFrame({
                    'segment': X_test, 'aspek': df_model.loc[X_test.index, 'aspek'],
                    'y_true': y_test, 'pred_nb': y_pred_nb, 'pred_svm': y_pred_svm
                })
                st.session_state['eval_data'] = eval_data
                
                st.success("Pelatihan selesai. Model disimpan di memori.")
                
        if 'eval_data' in st.session_state:
            eval_data = st.session_state['eval_data']
            st.subheader("Evaluasi Metrik Global (Data Uji 20%)")
            
            res = []
            for model_name, col_pred in [("Naive Bayes", "pred_nb"), ("Linear SVC", "pred_svm")]:
                y_t, y_p = eval_data['y_true'], eval_data[col_pred]
                res.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_t, y_p),
                    "Precision": precision_score(y_t, y_p, average='weighted', zero_division=0),
                    "Recall": recall_score(y_t, y_p, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_t, y_p, average='weighted', zero_division=0)
                })
            st.dataframe(pd.DataFrame(res).set_index("Model").style.format("{:.4f}"))
    else:
        st.warning("Lakukan Pelabelan di Tab 3 terlebih dahulu.")

# --- 5. EVALUASI DETAIL ---
elif menu == PAGES[4]:
    st.title("5. Evaluasi Per Aspek")
    
    if 'eval_data' in st.session_state:
        df_eval = st.session_state['eval_data']
        aspek_list = df_eval['aspek'].unique()
        
        res_aspek = []
        for a in aspek_list:
            sub = df_eval[df_eval['aspek'] == a]
            if len(sub) > 0:
                res_aspek.append({
                    "Aspek": a,
                    "Jumlah Uji": len(sub),
                    "Acc NB": accuracy_score(sub['y_true'], sub['pred_nb']),
                    "F1 NB": f1_score(sub['y_true'], sub['pred_nb'], average='weighted', zero_division=0),
                    "Acc SVM": accuracy_score(sub['y_true'], sub['pred_svm']),
                    "F1 SVM": f1_score(sub['y_true'], sub['pred_svm'], average='weighted', zero_division=0)
                })
        
        df_res_aspek = pd.DataFrame(res_aspek)
        st.dataframe(df_res_aspek.style.format({c: "{:.4f}" for c in df_res_aspek.columns if "Acc" in c or "F1" in c}))
        
        st.divider()
        st.subheader("Confusion Matrix Global")
        c1, c2 = st.columns(2)
        labels = sorted(df_eval['y_true'].unique())
        
        with c1:
            cm_nb = confusion_matrix(df_eval['y_true'], df_eval['pred_nb'], labels=labels)
            fig_nb, ax_nb = plt.subplots()
            sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_nb)
            ax_nb.set_title("Naive Bayes")
            st.pyplot(fig_nb)
            
        with c2:
            cm_svm = confusion_matrix(df_eval['y_true'], df_eval['pred_svm'], labels=labels)
            fig_svm, ax_svm = plt.subplots()
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=ax_svm)
            ax_svm.set_title("Linear SVC")
            st.pyplot(fig_svm)
    else:
        st.warning("Model belum dilatih. Masuk ke Tab 4.")

# --- 6. KLASIFIKASI MANUAL ---
elif menu == PAGES[5]:
    st.title("6. Klasifikasi Manual")
    
    if 'model_svm' in st.session_state and 'vectorizer' in st.session_state:
        nb_model = st.session_state['model_nb']
        svm_model = st.session_state['model_svm']
        vec = st.session_state['vectorizer']
        
        user_input = st.text_area("Masukkan teks ulasan:", height=150)
        
        if st.button("Analisis Teks"):
            if user_input.strip():
                segments = process_pipeline(user_input)
                if not segments:
                    st.warning("Teks tidak mengandung kata bermakna setelah dibersihkan.")
                else:
                    X_vec = vec.transform(segments)
                    pred_nb = nb_model.predict(X_vec)
                    pred_svm = svm_model.predict(X_vec)
                    
                    results = []
                    for i, seg in enumerate(segments):
                        aspek = get_aspect(seg)
                        results.append({
                            "Segmen Bersih": seg,
                            "Aspek": aspek,
                            "Prediksi NB": pred_nb[i],
                            "Prediksi SVM": pred_svm[i]
                        })
                    st.table(pd.DataFrame(results))
            else:
                st.warning("Input tidak boleh kosong.")
    else:
        st.warning("Silakan latih model di Tab 4 terlebih dahulu.")