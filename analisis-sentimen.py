import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import requests
import joblib
import os
from io import BytesIO
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen MBG", layout="wide")

# ==========================================
# 1. FUNGSI CACHED
# ==========================================
@st.cache_resource
def load_resources():
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)
    
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    factory_stop = StopWordRemoverFactory()
    stopwords_indo = factory_stop.get_stop_words()
    
    negation_words = {'tidak', 'tak', 'tiada', 'bukan', 'jangan', 'belum', 'kurang', 'gak', 'ga', 'nggak', 'enggak'}
    final_stopwords = set(stopwords_indo) - negation_words
    
    return stemmer, final_stopwords, negation_words

stemmer, final_stopwords, negation_words = load_resources()

@st.cache_data
def load_normalization_dict():
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/blob/main/kamuskatabaku.xlsx"
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
        topic_stopwords = {'mbg', 'makan', 'bergizi', 'gratis', 'gizi', 'program','prabowo', 'jokowi', 'presiden', 'menteri', 'indonesia'}
        for w in topic_stopwords:
            if w in lexicon: del lexicon[w]
        return lexicon
    except Exception:
        return {}

lexicon = load_inset_lexicon()

# ==========================================
# 2. FUNGSI UTAMA (Segmentasi & Preprocessing)
# ==========================================
def clean_text(text):
    """Tahap 1 & 2: Cleaning dan Case Folding"""
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)
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
    konjungsi = r'\b(tetapi|namun|dan|karena|meskipun|tapi|sedangkan)\b'
    # Hapus spasi kosong dan filter kata konjungsi itu sendiri agar tidak menjadi segmen terpisah
    segmen = [s.strip() for s in re.split(konjungsi, text) if s.strip() and s.strip() not in ['tetapi', 'namun', 'meskipun', 'tapi', 'sedangkan', 'cuman', 'cuma', 'sayangnya', 'padahal', 'walau', 'walaupun', 'pasalnya']]
    return segmen if segmen else [text]

def stopword_and_stem(text):
    """Tahap 6 & 7: Stopword Removal dan Stemming"""
    words = text.split()
    words = [w for w in words if w not in final_stopwords]
    return stemmer.stem(' '.join(words))

def preprocess_text(text):
    """
    Pipeline baru:
    1. Cleaning & Case Folding
    2. Tokenisasi & Normalisasi
    3. Segmentasi
    4. Stopword & Stemming (per segmen)
    """
    # 1. Cleaning & Case Folding
    text_clean = clean_text(text)
    
    # 2. Tokenisasi & Normalisasi
    text_norm = normalize_text(text_clean)
    
    # 3. Segmentasi
    segmen_list = segmentasi_kalimat(text_norm)
    
    # 4. Stopword & Stemming (per segmen)
    final_segments = []
    for seg in segmen_list:
        processed = stopword_and_stem(seg)
        if processed:
            final_segments.append(processed)
            
    return final_segments

def get_aspects(text):
    aspects = []
    text_words = set(str(text).split())
    keywords = {
        'Kualitas': ['kualitas', 'bagus', 'jelek', 'enak', 'basi', 'gizi', 'susu', 'menu', 'rasa', 'porsi', 'higienis', 'keracunan', 'sehat', 'mentah', 'keras', 'hambar', 'ulat', 'lauk', 'sayur', 'karbohidrat', 'protein', 'lemak', 'gula', 'ayam', 'telur', 'kenyang', 'alergi', 'higienitas'],
        'Layanan': ['layan', 'antri', 'ramah', 'lambat', 'cepat', 'bantu', 'saji', 'distribusi', 'vendor', 'katering', 'sekolah', 'siswa', 'guru', 'telat', 'molor', 'bocor', 'tepat waktu', 'pelosok', 'merata', 'zonasi', 'umkm', 'kemasan', 'kotak', 'plastik'],
        'Anggaran': ['harga', 'mahal', 'murah', 'biaya', 'bayar', 'anggar', 'boros', 'korupsi', 'dana', 'apbn', 'pajak', 'potong', 'sunat', 'markup', 'tender', 'proyek', 'apbd', 'defisit', 'utang', 'ekonomi', 'alokasi', 'transparan']
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

# --- TAB 1: UPLOAD ---
if menu == "1. Upload Data":
    st.header("Upload Dataset CSV")
    
    data_type = st.radio(
        "Pilih Jenis Data yang Diupload:",
        ("Data Mentah (Belum Preprocessing)", "Data Hasil Preprocessing (Skip ke Labeling)")
    )
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if data_type == "Data Mentah (Belum Preprocessing)":
            st.session_state['df_raw'] = df
            st.success(f"Data mentah berhasil dimuat! Jumlah baris dokumen: {len(df)}")
            st.dataframe(df.head())
            st.button("Lanjut ke Preprocessing 👉", on_click=set_page, args=(PAGES[1],))
        else:
            if 'processed_text' not in df.columns:
                st.error("File CSV tidak valid! Pastikan terdapat kolom 'processed_text' pada data Anda (unduh dari Tab 2 sebelumnya).")
            else:
                st.session_state['df_exploded'] = df
                st.success(f"Data hasil preprocessing berhasil dimuat! Jumlah baris segmen: {len(df)}")
                st.dataframe(df.head())
                st.button("Lanjut ke Labeling 👉", on_click=set_page, args=(PAGES[2],))

# --- TAB 2: PREPROCESSING ---
elif menu == "2. Preprocessing & Segmentasi":
    st.header("Preprocessing & Segmentasi Berbasis Konjungsi")
    st.warning("PERHATIAN: 1 baris teks dapat pecah menjadi beberapa baris segmen berdasarkan konjungsi (dan, tetapi, namun).")
    
    if st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw'].copy()
        col_name = st.selectbox("Pilih kolom teks:", df.columns)
        
        if st.button("Mulai Pemrosesan (Semua Data)"):
            with st.spinner('Memecah segmen dan membersihkan teks (Proses Stemming Sastrawi membutuhkan waktu)...'):
                # Simpan index asli untuk melacak dari dokumen mana segmen ini berasal
                df['doc_id'] = df.index
                
                # Apply preprocessing yang mereturn list of segments
                df['segmen_list'] = df[col_name].apply(preprocess_text)
                
                # EXPLODE: Meledakkan list menjadi baris-baris terpisah
                df_exploded = df.explode('segmen_list').dropna(subset=['segmen_list'])
                df_exploded.rename(columns={'segmen_list': 'processed_text'}, inplace=True)
                
                # Filter out segmen yang kosong setelah dibersihkan
                df_exploded = df_exploded[df_exploded['processed_text'].str.strip() != '']
                df_exploded = df_exploded.reset_index(drop=True)
                
                st.session_state['df_exploded'] = df_exploded
            
            st.success(f"Selesai! {len(df)} dokumen awal telah dipecah menjadi {len(df_exploded)} segmen independen.")
            st.dataframe(df_exploded[['doc_id', col_name, 'processed_text']].head(10))
        
        if st.session_state['df_exploded'] is not None:
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
        st.warning("Upload data terlebih dahulu.")

# --- TAB 3: LABELING ---
elif menu == "3. Labeling & Aspek":
    st.header("Pelabelan & Ekstraksi Aspek (Level Segmen)")
    
    if st.session_state['df_exploded'] is not None:
        df = st.session_state['df_exploded']
        
        if st.button("Jalankan Pelabelan & Aspek"):
            with st.spinner('Menentukan aspek dan sentimen pada setiap segmen...'):
                df['sentiment_label'] = df['processed_text'].apply(determine_sentiment)
                df['aspect_list'] = df['processed_text'].apply(get_aspects)
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
                
            st.dataframe(df[['processed_text', 'sentiment_label', 'aspect_list']].head())
            
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
                        
                        cols_to_show = ['processed_text', 'sentiment_label']
                        # Mencari kolom teks asli untuk ditampilkan (kolom selain doc_id dan hasil proses)
                        other_cols = [c for c in subset_aspek.columns if c not in ['doc_id', 'processed_text', 'sentiment_label', 'aspect_list']]
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
                    all_words = " ".join(subset_data['processed_text'].astype(str)).split()
                    
                    if all_words:
                        freq_data = pd.Series(all_words).value_counts().head(10)
                        fig_freq, ax_freq = plt.subplots(figsize=(5, 6))
                        sns.barplot(x=freq_data.values, y=freq_data.index, ax=ax_freq, palette='viridis', hue=freq_data.index, legend=False)
                        ax_freq.set_xlabel("Frekuensi")
                        st.pyplot(fig_freq)
                    else:
                        st.info("Tidak ada data kata.")
            
            # --- VISUALISASI WORDCLOUD ---
            st.divider()
            st.subheader("Visualisasi WordCloud")
            wc_c1, wc_c2, wc_c3 = st.columns(3)
            
            with wc_c1:
                st.markdown("**Sentimen Positif**")
                text_pos = " ".join(df[df['sentiment_label'] == 'Positif']['processed_text'].astype(str))
                if text_pos.strip():
                    wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(text_pos)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_pos, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
            
            with wc_c2:
                st.markdown("**Sentimen Negatif**")
                text_neg = " ".join(df[df['sentiment_label'] == 'Negatif']['processed_text'].astype(str))
                if text_neg.strip():
                    wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(text_neg)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_neg, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
            
            with wc_c3:
                st.markdown("**Sentimen Netral**")
                text_neu = " ".join(df[df['sentiment_label'] == 'Netral']['processed_text'].astype(str))
                if text_neu.strip():
                    wc_neu = WordCloud(width=400, height=300, background_color='white', colormap='Blues').generate(text_neu)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc_neu, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
        
        if 'sentiment_label' in df.columns:
            st.button("Lanjut ke Modeling 👉", on_click=set_page, args=(PAGES[3],))
    else:
        st.warning("Lakukan Preprocessing & Segmentasi terlebih dahulu.")

# --- TAB 4: MODELING ---
elif menu == "4. Modeling (Training)":
    st.header("Training & Evaluasi Model (Level Segmen)")
    
    if st.session_state['df_exploded'] is not None and 'sentiment_label' in st.session_state['df_exploded'].columns:
        df = st.session_state['df_exploded']
        
        remove_neutral = st.checkbox("Hapus Data Netral?", value=True)
        # GANTI CHECKBOX BALANCING LAMA DENGAN SMOTE
        use_smote = st.checkbox("Gunakan SMOTE untuk menyeimbangkan data? (Sangat Direkomendasikan)", value=True)
        
        df_model = df[df['sentiment_label'] != 'Netral'] if remove_neutral else df
            
        st.subheader("Pilih Metode Modeling")
        model_action = st.radio("Aksi:", ["Latih Model Baru", "Muat Model Tersimpan"], horizontal=True)
        
        if model_action == "Latih Model Baru":
            if st.button("Mulai Training Model"):
                with st.spinner("Melatih model dengan SMOTE & GridSearchCV... (Ini membutuhkan waktu lebih lama untuk mencari parameter terbaik)"):
                    
                    X = df_model['processed_text']
                    y = df_model['sentiment_label']
                    
                    # Split Data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    
                    # --- SETUP PARAMETER UNTUK GRIDSEARCH ---
                    param_grid_nb = {
                        'clf__alpha': [0.1, 0.5, 1.0, 2.0]
                    }
                    param_grid_svm = {
                        'clf__C': [0.1, 1, 10], 
                        'clf__kernel': ['linear', 'rbf'] # Biarkan komputer mencari mana yang lebih baik
                    }

                    # --- SETUP PIPELINE BERDASARKAN PILIHAN SMOTE ---
                    if use_smote:
                        # Jika pakai SMOTE, wajib pakai ImbPipeline, BUKAN sklearn Pipeline
                        pipe_nb = ImbPipeline([
                            ('tfidf', TfidfVectorizer()), 
                            ('smote', SMOTE(random_state=42)), 
                            ('clf', MultinomialNB())
                        ])
                        pipe_svm = ImbPipeline([
                            ('tfidf', TfidfVectorizer()), 
                            ('smote', SMOTE(random_state=42)), 
                            ('clf', SVC(probability=True, random_state=42))
                        ])
                    else:
                        pipe_nb = ImbPipeline([
                            ('tfidf', TfidfVectorizer()), 
                            ('clf', MultinomialNB())
                        ])
                        pipe_svm = ImbPipeline([
                            ('tfidf', TfidfVectorizer()), 
                            ('clf', SVC(probability=True, random_state=42))
                        ])
                    
                    # --- EKSEKUSI GRIDSEARCHCV ---
                    grid_nb = GridSearchCV(pipe_nb, param_grid_nb, cv=3, n_jobs=-1, scoring='accuracy')
                    grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=3, n_jobs=-1, scoring='accuracy')
                    
                    grid_nb.fit(X_train, y_train)
                    grid_svm.fit(X_train, y_train)
                    
                    # --- AMBIL MODEL TERBAIK ---
                    best_nb = grid_nb.best_estimator_
                    best_svm = grid_svm.best_estimator_
                    
                    y_pred_nb = best_nb.predict(X_test)
                    y_pred_svm = best_svm.predict(X_test)
                    
                    # --- HITUNG TF-IDF UNTUK VISUALISASI ---
                    vectorizer = best_nb.named_steps['tfidf']
                    X_tfidf = vectorizer.transform(X_train)
                    sum_tfidf = X_tfidf.sum(axis=0)
                    words = vectorizer.get_feature_names_out()
                    tfidf_data = [(words[i], sum_tfidf[0, i]) for i in range(len(words))]
                    df_tfidf = pd.DataFrame(tfidf_data, columns=['Kata', 'Skor Total TF-IDF'])
                    df_tfidf = df_tfidf.sort_values(by='Skor Total TF-IDF', ascending=False).head(10).reset_index(drop=True)
                    
                    # --- SIMPAN KE SESSION STATE ---
                    st.session_state['model_nb'] = best_nb
                    st.session_state['model_svm'] = best_svm
                    
                    st.session_state['y_test'] = y_test
                    st.session_state['y_pred_nb'] = y_pred_nb
                    st.session_state['y_pred_svm'] = y_pred_svm
                    st.session_state['df_tfidf'] = df_tfidf
                    
                    test_df = df_model.loc[X_test.index].copy()
                    test_df['y_true'] = y_test
                    test_df['pred_nb'] = y_pred_nb
                    test_df['pred_svm'] = y_pred_svm
                    st.session_state['test_data_eval'] = test_df
                    
                    # --- SIMPAN MODEL KE DISK ---
                    saved_data = {
                        'model_nb': best_nb,
                        'model_svm': best_svm,
                        'y_test': y_test,
                        'y_pred_nb': y_pred_nb,
                        'y_pred_svm': y_pred_svm,
                        'test_data_eval': test_df,
                        'best_params_nb': grid_nb.best_params_,
                        'best_params_svm': grid_svm.best_params_,
                        'df_tfidf': df_tfidf
                    }
                    try:
                        joblib.dump(saved_data, 'saved_model_data.joblib')
                        st.success("Training dan Optimasi Selesai! Model berhasil disimpan.")
                    except Exception as e:
                        st.warning(f"Model dilatih namun gagal disimpan: {e}")
                    
                st.info(f"✨ Parameter Naive Bayes Terbaik: {grid_nb.best_params_}")
                st.info(f"✨ Parameter SVM Terbaik: {grid_svm.best_params_}")
        else:
            if st.button("Muat Model Tersimpan"):
                with st.spinner("Memuat model dari penyimpanan..."):
                    if os.path.exists('saved_model_data.joblib'):
                        saved_data = joblib.load('saved_model_data.joblib')
                        st.session_state['model_nb'] = saved_data['model_nb']
                        st.session_state['model_svm'] = saved_data['model_svm']
                        st.session_state['y_test'] = saved_data['y_test']
                        st.session_state['y_pred_nb'] = saved_data['y_pred_nb']
                        st.session_state['y_pred_svm'] = saved_data['y_pred_svm']
                        st.session_state['test_data_eval'] = saved_data['test_data_eval']
                        if 'df_tfidf' in saved_data:
                            st.session_state['df_tfidf'] = saved_data['df_tfidf']
                            
                        st.success("Model berhasil dimuat dari penyimpanan!")
                        if 'best_params_nb' in saved_data:
                            st.info(f"✨ Parameter Naive Bayes: {saved_data['best_params_nb']}")
                        if 'best_params_svm' in saved_data:
                            st.info(f"✨ Parameter SVM: {saved_data['best_params_svm']}")
                    else:
                        st.error("Tidak ditemukan file model yang tersimpan ('saved_model_data.joblib'). Silakan pilih 'Latih Model Baru' terlebih dahulu.")
        
        st.divider()
        
        if 'y_test' in st.session_state and 'model_nb' in st.session_state and 'model_svm' in st.session_state:
            st.subheader("1. Evaluasi Kinerja Model (Data Testing 20%)")
            
            y_test = st.session_state['y_test']
            y_pred_nb = st.session_state['y_pred_nb']
            y_pred_svm = st.session_state['y_pred_svm']
            
            # Matriks Evaluasi (Berdasarkan CV)
            def get_metrics(y_true, y_pred, model_name):
                return {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }

            metrics_nb = get_metrics(y_test, y_pred_nb, "Naive Bayes")
            metrics_svm = get_metrics(y_test, y_pred_svm, "SVM")
            st.dataframe(pd.DataFrame([metrics_nb, metrics_svm]).set_index("Model").style.highlight_max(axis=0, color='green').format("{:.2%}"))
            
            # Confusion Matrix (Berdasarkan CV)
            labels = sorted(pd.concat([y_test, pd.Series(y_pred_nb), pd.Series(y_pred_svm)]).unique())

            fig_cv, ax_cv = plt.subplots(1, 2, figsize=(12, 5))
            cm_nb_cv = confusion_matrix(y_test, y_pred_nb, labels=labels)
            sns.heatmap(cm_nb_cv, annot=True, fmt='d', cmap='Blues', ax=ax_cv[0], xticklabels=labels, yticklabels=labels)
            ax_cv[0].set_title(f"Confusion Matrix Naive Bayes")
            ax_cv[0].set_xlabel("Predicted")
            ax_cv[0].set_ylabel("Actual")

            cm_svm_cv = confusion_matrix(y_test, y_pred_svm, labels=labels)
            sns.heatmap(cm_svm_cv, annot=True, fmt='d', cmap='Greens', ax=ax_cv[1], xticklabels=labels, yticklabels=labels)
            ax_cv[1].set_title(f"Confusion Matrix SVM")
            ax_cv[1].set_xlabel("Predicted")
            ax_cv[1].set_ylabel("Actual")
            st.pyplot(fig_cv)
            
            # --- TF-IDF VISUALIZATION ---
            st.divider()
            st.subheader("3. Kata dengan Bobot TF-IDF Tertinggi")
            
            if 'df_tfidf' in st.session_state:
                df_tfidf = st.session_state['df_tfidf']
                c_tf1, c_tf2 = st.columns([1, 2])
                with c_tf1:
                    st.dataframe(df_tfidf)
                with c_tf2:
                    fig_tf, ax_tf = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='Skor Total TF-IDF', y='Kata', data=df_tfidf, palette='viridis', ax=ax_tf)
                    ax_tf.set_title("Top 10 Kata Paling Berbobot (TF-IDF)")
                    st.pyplot(fig_tf)
            else:
                st.info("Data TF-IDF tidak tersedia di session state.")

            # --- PROBABILITAS PRIOR & LIKELIHOOD NAIVE BAYES ---
            st.divider()
            st.subheader("4. Probabilitas Kelas (Prior) & Likelihood - Naive Bayes")
            
            # Ekstrak model dan vectorizer dari pipeline
            nb_model = st.session_state['model_nb'].named_steps['clf']
            tfidf_vectorizer = st.session_state['model_nb'].named_steps['tfidf']
            
            # 1. Menarik nilai Prior Probability
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
            
            svm_model = st.session_state['model_svm'].named_steps['clf']
            tfidf_vectorizer_svm = st.session_state['model_svm'].named_steps['tfidf']
            
            st.write(f"**Kernel SVM Hasil Evaluasi:** `{svm_model.kernel}`")
            
            if svm_model.kernel == 'linear':
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
            else:
                st.warning("⚠️ Karena GridSearch (atau setelan model) memilih kernel non-linear, analisis Bobot Kata (Weights) tidak dapat ditampilkan. Model non-linear tidak memiliki koefisien kata tunggal (coef_) seperti regresi linear.")

            if 'model_svm' in st.session_state:
                st.button("Lanjut ke Evaluasi 👉", on_click=set_page, args=(PAGES[4],))
        else:
            st.info("Silakan latih model baru atau muat model tersimpan untuk melihat hasil evaluasi.")
    else:
        st.warning("Data belum dilabeli.")

# --- TAB 5: EVALUASI PER ASPEK ---
elif menu == "5. Evaluasi Detail (Per Aspek)":
    st.header("Evaluasi Performa Model (Global & Per Aspek)")
    
    if 'test_data_eval' in st.session_state:
        df_eval = st.session_state['test_data_eval']
        
        # --- 1. EVALUASI GLOBAL ---
        st.subheader("1. Evaluasi Global (Data Testing 20%)")
        
        def calculate_metrics(y_true, y_pred, model_name):
            return {
                "Model": model_name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }

        global_nb = calculate_metrics(df_eval['y_true'], df_eval['pred_nb'], "Naive Bayes")
        global_svm = calculate_metrics(df_eval['y_true'], df_eval['pred_svm'], "SVM")
        
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
        
        st.divider()

        # --- 2. EVALUASI PER ASPEK ---
        st.subheader("2. Evaluasi Detail per Aspek")
        df_exploded = df_eval.explode('aspect_list')
        
        aspect_metrics = []
        for aspect in df_exploded['aspect_list'].unique():
            subset = df_exploded[df_exploded['aspect_list'] == aspect]
            if len(subset) > 0:
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
        
        fig_aspect, ax_aspect = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_df, x='Aspek', y='Akurasi', hue='Model', palette='coolwarm', ax=ax_aspect)
        ax_aspect.set_ylim(0, 1.1)
        ax_aspect.set_title("Perbandingan Akurasi NB vs SVM per Aspek")
        plt.xticks(rotation=45)
        for p in ax_aspect.patches:
             if p.get_height() > 0:
                ax_aspect.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8)
        st.pyplot(fig_aspect)
        
        # --- WORDCLOUD PER ASPEK ---
        st.divider()
        st.subheader("Visualisasi WordCloud per Aspek")
        aspects_list = metrics_df['Aspek'].tolist()
        if aspects_list:
            tabs = st.tabs(aspects_list)
            for i, aspect in enumerate(aspects_list):
                with tabs[i]:
                    subset = df_exploded[df_exploded['aspect_list'] == aspect]
                    
                    wc_col1, wc_col2, wc_col3 = st.columns(3)
                    
                    # POSITIF
                    with wc_col1:
                        st.markdown("##### Sentimen Positif")
                        subset_pos = subset[subset['sentiment_label'] == 'Positif']
                        text_pos = " ".join(subset_pos['processed_text'].astype(str))
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
                        text_neg = " ".join(subset_neg['processed_text'].astype(str))
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
                        text_neu = " ".join(subset_neu['processed_text'].astype(str))
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

# --- TAB 6: PREDIKSI ---
# --- TAB 6: PREDIKSI MANUAL ---
elif menu == "6. Prediksi Manual":
    st.header("Uji Coba Prediksi Manual")
    
    # 1. VALIDASI LOGIS: Cek apakah model di Tab 4 sudah dilatih
    if 'model_nb' not in st.session_state or 'model_svm' not in st.session_state:
        st.error("Model belum dilatih! Anda tidak bisa melakukan prediksi manual.")
        st.warning("Silakan kembali ke 'Tab 4. Modeling', jalankan training, lalu kembali ke sini.")
    else:
        st.info("Menggunakan model Naive Bayes dan SVM yang telah dilatih. Prediksi dilakukan per segmen kalimat.")
        
        # Ambil model (Pipeline) dari memori
        model_nb = st.session_state['model_nb']
        model_svm = st.session_state['model_svm']
        
        # Area input user
        user_input = st.text_area("Masukkan teks atau tweet yang ingin diprediksi sentimennya:", height=150)
        
        if st.button("Prediksi Sentimen"):
            if user_input.strip() == "":
                st.warning("Teks tidak boleh kosong. Masukkan kalimat untuk diuji.")
            else:
                with st.spinner("Memproses teks dan melakukan prediksi..."):
                    # `preprocess_text` mengembalikan list dari segmen yang sudah diproses.
                    processed_segments = preprocess_text(user_input)
                    
                    if not processed_segments:
                        st.warning("Teks tidak menghasilkan segmen yang dapat dianalisis setelah preprocessing.")
                        st.stop()

                    # Lakukan prediksi untuk setiap segmen.
                    preds_nb = model_nb.predict(processed_segments)
                    preds_svm = model_svm.predict(processed_segments)
                    
                    st.divider()
                    st.subheader("Hasil Analisis")
                    
                    st.markdown("**Teks Asli:**")
                    st.write(f"> *{user_input}*")
                    
                    st.markdown("**Hasil Prediksi per Segmen:**")
                    
                    def get_color(label):
                        if str(label).lower() == 'positif': return "🟢"
                        elif str(label).lower() == 'negatif': return "🔴"
                        return "⚪"
                    
                    # Buat DataFrame untuk menampilkan hasil dengan rapi
                    results_data = []
                    for i, segment in enumerate(processed_segments):
                        aspects = get_aspects(segment)
                        results_data.append({
                            "Segmen Teks": segment,
                            "Aspek": ", ".join(aspects),
                            "Prediksi Naive Bayes": f"{get_color(preds_nb[i])} {preds_nb[i]}",
                            "Prediksi SVM": f"{get_color(preds_svm[i])} {preds_svm[i]}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.table(results_df)