# 📊 Aspect-Based Sentiment Analysis (ABSA) - Program Makan Bergizi Gratis (MBG)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)

Repositori ini berisi *source code* aplikasi penelitian skripsi yang berfokus pada **Analisis Sentimen Berbasis Aspek (ABSA)** terhadap opini publik terkait **Program Makan Bergizi Gratis (MBG)**. 

Penelitian ini membandingkan kinerja dua algoritma *Machine Learning*, yaitu **Multinomial Naive Bayes** dan **Linear Support Vector Classifier (LinearSVC)**, dengan menerapkan teknik unik berupa **segmentasi kalimat berbasis konjungsi** untuk memisahkan opini majemuk sebelum dilakukan klasifikasi.

---

## ✨ Fitur Utama (Web App Features)

Aplikasi dibangun menggunakan **Streamlit** dan dibagi menjadi 6 tahapan pemrosesan end-to-end:

1. **📂 Upload Data:** Mendukung manajemen dataset mentah maupun data yang sudah melalui tahap *preprocessing*.
2. **⚙️ Preprocessing & Segmentasi:** Mengimplementasikan *Cleaning*, *Case Folding*, *Normalization* (Kamus Kata Baku), *Stopword Removal*, *Stemming* (Sastrawi), dan pemecahan kalimat majemuk menggunakan kata hubung (konjungsi).
3. **🏷️ Labeling & Ekstraksi Aspek:** Menggunakan pendekatan hibrida (*InSet Lexicon* untuk sentimen dan *Rule-based keywords* untuk ekstraksi aspek: **Kualitas, Layanan, Anggaran**).
4. **🤖 Modeling (Training):** Pelatihan model secara *real-time* dengan ekstraksi fitur **TF-IDF**. Dilengkapi antarmuka diagnostik mendalam untuk membedah otak model (melihat *Prior*, *Likelihood* Naive Bayes, dan *Decision Function/Weights* SVM).
5. **📈 Evaluasi Detail:** Menampilkan metrik evaluasi global dan per aspek (*Accuracy, Precision, Recall, F1-Score*), *Confusion Matrix*, dan visualisasi *WordCloud*.
6. **📝 Klasifikasi Manual:** Modul pengujian model menggunakan teks atau opini baru yang diinputkan pengguna.

---

## 🛠️ Teknologi & Library yang Digunakan

* **Bahasa Pemrograman:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (`MultinomialNB`, `LinearSVC`, `TfidfVectorizer`)
* **Natural Language Processing (NLP):** NLTK, Sastrawi (Stemmer & Stopword)
* **Data Manipulation & Math:** Pandas, Numpy
* **Data Visualization:** Matplotlib, Seaborn, WordCloud

---

## 🚀 Cara Instalasi dan Menjalankan Aplikasi

Pastikan Anda telah menginstal [Python](https://www.python.org/downloads/) di sistem operasi Anda. Berikut adalah langkah-langkah untuk menjalankan aplikasi ini di lingkungan lokal (Windows):

**1. Clone Repositori**
```cmd
git clone [https://github.com/rifqieali/analisis_sentimen_MBG.git](https://github.com/rifqieali/analisis_sentimen_MBG.git)
cd analisis_sentimen_MBG 
```
**2. Buat Virtual Environment (Sangat Direkomendasikan)**

```cmd
python -m venv venv
venv\Scripts\activate
```
**3. Instal Dependensi**

```cmd
pip install streamlit pandas numpy scikit-learn nltk Sastrawi matplotlib seaborn wordcloud joblib openpyxl requests
```
**4. Jalankan Aplikasi Streamlit**
(Ganti absa_final.py dengan nama file utama Python Anda jika berbeda)

```cmd
streamlit run absa_final.py
```
📂 Struktur Direktori (Bisa Disesuaikan)
```Plaintext
analisis_sentimen_MBG/
│
├── Data/                   # Folder berisi dataset CSV (mentah & hasil preprocessing)
├── Hasil/                  # Folder hasil export gambar/grafik (jika ada)
├── absa_final.py           # File utama aplikasi Streamlit
├── saved_model_data.joblib # File backup model Machine Learning (di-generate otomatis)
├── .gitignore              # Daftar file/folder yang diabaikan Git (misal: /venv)
└── README.md               # Dokumentasi repositori
