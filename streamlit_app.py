import requests
from bs4 import BeautifulSoup
import streamlit as st
import re
import pandas as pd
import json
from datetime import datetime
from urllib.parse import urljoin, quote_plus
import hashlib
import time
import os
import base64
import io
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, pipeline
import torch

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==================== KONFIGURASI GITHUB ====================
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")  # Token dari Streamlit secrets
GITHUB_REPO = "abdfajar/republika_sentiner"
GITHUB_BRANCH = "main"
SCRAPPER_RESULT_PATH = "scrapper_result"
ANALYSIS_PATH = "analisis"

# ==================== KONFIGURASI MODEL ====================
SENTIMENT_MODEL_NAME = "indolem/indobert-base-uncased"
NER_MODEL_NAME = "cahya/bert-base-indonesian-NER"

# Cache untuk model
@st.cache_resource
def load_sentiment_model():
    """Load model sentiment analysis"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

@st.cache_resource
def load_ner_model():
    """Load model NER"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        return ner_pipeline
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

# ==================== FUNGSI ANALISIS TEKS ====================
def preprocess_text(text):
    """Preprocessing teks untuk analisis"""
    if not text or pd.isna(text):
        return ""
    
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    
    return text

def predict_sentiment(texts, sentiment_pipeline):
    """Prediksi sentimen untuk list of texts"""
    if not sentiment_pipeline:
        return []
    
    results = []
    for text in texts:
        if not text or len(text.strip()) < 10:
            results.append({"label": "NETRAL", "score": 0.5})
            continue
        
        try:
            # Truncate text jika terlalu panjang
            truncated_text = text[:512]
            result = sentiment_pipeline(truncated_text)[0]
            
            # Map label ke bahasa Indonesia
            label_map = {
                "LABEL_0": "NEGATIF",
                "LABEL_1": "POSITIF", 
                "LABEL_2": "NETRAL"
            }
            
            result["label"] = label_map.get(result["label"], result["label"])
            results.append(result)
            
        except Exception as e:
            st.warning(f"Error predicting sentiment: {e}")
            results.append({"label": "NETRAL", "score": 0.5})
    
    return results

def extract_entities(texts, ner_pipeline):
    """Extract named entities dari list of texts"""
    if not ner_pipeline:
        return []
    
    all_entities = []
    
    for text in texts:
        if not text or len(text.strip()) < 10:
            all_entities.append([])
            continue
        
        try:
            # Truncate text jika terlalu panjang
            truncated_text = text[:512]
            entities = ner_pipeline(truncated_text)
            
            # Filter entities dengan score tinggi
            filtered_entities = [
                {
                    "entity": entity["word"],
                    "type": entity["entity_group"],
                    "score": entity["score"]
                }
                for entity in entities if entity["score"] > 0.8
            ]
            
            all_entities.append(filtered_entities)
            
        except Exception as e:
            st.warning(f"Error extracting entities: {e}")
            all_entities.append([])
    
    return all_entities

def extract_trigrams(texts):
    """Extract trigrams dari list of texts"""
    all_trigrams = []
    
    for text in texts:
        if not text:
            all_trigrams.append([])
            continue
        
        try:
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Generate trigrams
            trigrams_list = list(ngrams(tokens, 3))
            
            # Convert to string
            trigram_strings = [' '.join(trigram) for trigram in trigrams_list]
            all_trigrams.append(trigram_strings)
            
        except Exception as e:
            st.warning(f"Error extracting trigrams: {e}")
            all_trigrams.append([])
    
    return all_trigrams

def calculate_tfidf(texts):
    """Calculate TF-IDF scores untuk list of texts"""
    if not texts:
        return [], []
    
    try:
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        valid_texts = [text for text in processed_texts if text and len(text.strip()) > 10]
        
        if not valid_texts:
            return [], []
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words=None,  # Tidak menggunakan stopwords karena bahasa Indonesia
            ngram_range=(1, 2)  # Unigrams dan bigrams
        )
        
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores across all documents
        tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Create sorted list of (word, score) pairs
        word_scores = list(zip(feature_names, tfidf_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores, feature_names
        
    except Exception as e:
        st.warning(f"Error calculating TF-IDF: {e}")
        return [], []

# ==================== FUNGSI ANALISIS SENTIMEN UTAMA ====================
def analyze_sentiment_comprehensive(start_date, end_date):
    """Fungsi utama untuk analisis sentimen yang komprehensif"""
    
    # Load data yang difilter
    keyword_df, results_df, metadata_df = load_and_filter_data(start_date, end_date)
    
    if metadata_df.empty:
        st.warning("Tidak ada data untuk dianalisis")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Load models
    with st.spinner("Memuat model AI..."):
        sentiment_pipeline = load_sentiment_model()
        ner_pipeline = load_ner_model()
    
    if not sentiment_pipeline:
        st.error("Gagal memuat model sentiment analysis")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Prepare texts for analysis
    texts = []
    titles = []
    
    for idx, row in metadata_df.iterrows():
        title = row.get('judul', '')
        content = row.get('konten', '')
        
        # Gabungkan judul dan konten untuk analisis
        combined_text = f"{title}. {content}" if title else content
        texts.append(combined_text)
        titles.append(title)
    
    # 1. PREDIKSI SENTIMEN
    st.info("🔍 Melakukan prediksi sentimen...")
    sentiment_results = predict_sentiment(texts, sentiment_pipeline)
    
    sentiment_data = []
    for i, (title, result) in enumerate(zip(titles, sentiment_results)):
        sentiment_data.append({
            'judul': title[:100] + '...' if len(title) > 100 else title,
            'sentimen': result.get('label', 'NETRAL'),
            'skor_kepercayaan': round(result.get('score', 0.5), 3)
        })
    
    df_sentiment = pd.DataFrame(sentiment_data)
    
    # 2. NAMED ENTITY RECOGNITION (NER)
    st.info("🏷️ Melakukan ekstraksi entitas...")
    entities_results = extract_entities(texts, ner_pipeline)
    
    # Aggregate entities across all documents
    entity_counter = Counter()
    type_counter = Counter()
    
    for entities in entities_results:
        for entity in entities:
            entity_name = entity['entity']
            entity_type = entity['type']
            entity_counter[entity_name] += 1
            type_counter[entity_type] += 1
    
    # Prepare NER results
    ner_data = []
    for entity, count in entity_counter.most_common(20):  # Top 20 entities
        ner_data.append({
            'entity': entity,
            'tipe': 'ENTITAS',
            'frekuensi': count
        })
    
    df_ner = pd.DataFrame(ner_data)
    
    # 3. TRIGRAM ANALYSIS
    st.info("🔤 Melakukan analisis trigram...")
    trigrams_results = extract_trigrams(texts)
    
    # Aggregate trigrams across all documents
    trigram_counter = Counter()
    for trigrams in trigrams_results:
        for trigram in trigrams:
            trigram_counter[trigram] += 1
    
    # Prepare trigram results
    trigram_data = []
    for trigram, count in trigram_counter.most_common(15):  # Top 15 trigrams
        trigram_data.append({
            'trigram': trigram,
            'frekuensi': count
        })
    
    df_trigram = pd.DataFrame(trigram_data)
    
    # 4. TF-IDF ANALYSIS
    st.info("📊 Menghitung TF-IDF...")
    tfidf_scores, feature_names = calculate_tfidf(texts)
    
    # Prepare TF-IDF results
    tfidf_data = []
    for word, score in tfidf_scores[:20]:  # Top 20 terms
        tfidf_data.append({
            'kata_kunci': word,
            'skor_tfidf': round(score, 4)
        })
    
    df_tfidf = pd.DataFrame(tfidf_data)
    
    # Simpan hasil analisis ke GitHub
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Upload hasil analisis ke folder analisis
        upload_to_github(
            f"{ANALYSIS_PATH}/sentiment_prediction_{timestamp}.csv",
            df_sentiment,
            f"Analisis sentimen {timestamp}"
        )
        
        upload_to_github(
            f"{ANALYSIS_PATH}/ner_results_{timestamp}.csv",
            df_ner,
            f"Analisis NER {timestamp}"
        )
        
        upload_to_github(
            f"{ANALYSIS_PATH}/trigram_results_{timestamp}.csv",
            df_trigram,
            f"Analisis trigram {timestamp}"
        )
        
        upload_to_github(
            f"{ANALYSIS_PATH}/tfidf_results_{timestamp}.csv",
            df_tfidf,
            f"Analisis TF-IDF {timestamp}"
        )
        
        # Simpan summary analisis
        summary_data = {
            'timestamp_analisis': timestamp,
            'total_dokumen': len(metadata_df),
            'sentimen_positif': len(df_sentiment[df_sentiment['sentimen'] == 'POSITIF']),
            'sentimen_negatif': len(df_sentiment[df_sentiment['sentimen'] == 'NEGATIF']),
            'sentimen_netral': len(df_sentiment[df_sentiment['sentimen'] == 'NETRAL']),
            'total_entitas': len(df_ner),
            'total_trigram': len(df_trigram),
            'rentang_tanggal': f"{start_date} to {end_date}"
        }
        
        df_summary = pd.DataFrame([summary_data])
        upload_to_github(
            f"{ANALYSIS_PATH}/analysis_summary_{timestamp}.csv",
            df_summary,
            f"Summary analisis {timestamp}"
        )
        
        st.success("✅ Hasil analisis berhasil disimpan ke GitHub")
        
    except Exception as e:
        st.error(f"❌ Gagal menyimpan hasil analisis ke GitHub: {str(e)}")
    
    return df_sentiment, df_ner, df_trigram, df_tfidf

# ==================== FUNGSI GITHUB ====================
def get_github_headers():
    """Mendapatkan headers untuk request GitHub API"""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def github_api_request(endpoint, method="GET", data=None):
    """Membuat request ke GitHub API"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/{endpoint}"
    headers = get_github_headers()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        
        response.raise_for_status()
        return response.json() if response.content else {}
    except Exception as e:
        st.error(f"Error GitHub API: {str(e)}")
        return None

def get_file_sha(file_path):
    """Mendapatkan SHA hash file yang ada di GitHub"""
    endpoint = f"contents/{file_path}?ref={GITHUB_BRANCH}"
    result = github_api_request(endpoint)
    return result.get("sha") if result else None

def upload_to_github(file_path, content, commit_message):
    """Upload file ke GitHub"""
    # Encode content to base64
    if isinstance(content, pd.DataFrame):
        content = content.to_csv(index=False)
    
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    content_b64 = base64.b64encode(content).decode('utf-8')
    
    # Get existing file SHA if exists
    sha = get_file_sha(file_path)
    
    data = {
        "message": commit_message,
        "content": content_b64,
        "branch": GITHUB_BRANCH
    }
    
    if sha:
        data["sha"] = sha
    
    endpoint = f"contents/{file_path}"
    result = github_api_request(endpoint, "PUT", data)
    return result is not None

def download_from_github(file_path):
    """Download file dari GitHub"""
    endpoint = f"contents/{file_path}?ref={GITHUB_BRANCH}"
    result = github_api_request(endpoint)
    
    if result and "content" in result:
        content_b64 = result["content"]
        content = base64.b64decode(content_b64).decode('utf-8')
        return content
    return None

def list_github_files(folder_path):
    """Mendapatkan daftar file di folder GitHub"""
    endpoint = f"contents/{folder_path}?ref={GITHUB_BRANCH}"
    result = github_api_request(endpoint)
    
    if result and isinstance(result, list):
        return [item["name"] for item in result if item["type"] == "file"]
    return []

def sync_to_github():
    """Sinkronisasi file lokal ke GitHub"""
    csv_dir = "scrapper_result"
    if not os.path.exists(csv_dir):
        st.warning("Folder scrapper_result tidak ditemukan secara lokal")
        return False
    
    success_count = 0
    total_count = 0
    
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_dir, filename)
            github_file_path = f"{SCRAPPER_RESULT_PATH}/{filename}"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if upload_to_github(github_file_path, content, f"Sync {filename}"):
                    success_count += 1
                    st.success(f"✅ {filename} berhasil diupload ke GitHub")
                else:
                    st.error(f"❌ Gagal upload {filename} ke GitHub")
                
                total_count += 1
                time.sleep(1)  # Delay untuk menghindari rate limit
                
            except Exception as e:
                st.error(f"❌ Error sync {filename}: {str(e)}")
    
    return success_count, total_count

def load_from_github():
    """Load file dari GitHub ke lokal"""
    csv_dir = "scrapper_result"
    os.makedirs(csv_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Load dari folder scrapper_result
    files = list_github_files(SCRAPPER_RESULT_PATH)
    
    for filename in files:
        if filename.endswith('.csv'):
            github_file_path = f"{SCRAPPER_RESULT_PATH}/{filename}"
            local_file_path = os.path.join(csv_dir, filename)
            
            try:
                content = download_from_github(github_file_path)
                if content:
                    with open(local_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    success_count += 1
                    st.success(f"✅ {filename} berhasil didownload dari GitHub")
                else:
                    st.error(f"❌ Gagal download {filename} dari GitHub")
                
                total_count += 1
                time.sleep(1)  # Delay untuk menghindari rate limit
                
            except Exception as e:
                st.error(f"❌ Error load {filename}: {str(e)}")
    
    return success_count, total_count

# ==================== KONFIGURASI STREAMLIT ====================
st.set_page_config(
    page_title="🔍 Republika.co.id Search Scraper",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FUNGSI SCRAPING ARTIKEL ====================
def clean_text(text):
    """Membersihkan teks dari karakter tidak diinginkan"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    return text.strip()

def extract_text_from_element(element):
    """Ekstrak teks dari elemen dengan pembersihan"""
    if not element:
        return ""
    element_copy = BeautifulSoup(str(element), 'html.parser')
    for unwanted in element_copy(['script', 'style', 'nav', 'header', 'footer', 'aside', 'figure', 'img', 'video', 'blockquote']):
        unwanted.decompose()
    text = element_copy.get_text(separator='\n', strip=True)
    return clean_text(text)

def extract_republika_article(url):
    """
    Fungsi utama untuk scraping artikel Republika.co.id
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        st.write(f"🔍 Mengakses URL: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        metadata = {
            'judul': '',
            'waktu_terbit': '',
            'editor': '',
            'konten': '',
            'url': url,
            'panjang_konten': 0
        }
        main_content = soup.find('div', class_='main-content__left')
        if not main_content:
            return None, "Struktur halaman tidak dikenali. Tidak ditemukan div.main-content__left"
        title_div = main_content.find('div', class_='max-card__title')
        if title_div:
            title_h1 = title_div.find('h1')
            metadata['judul'] = clean_text(title_h1.get_text()) if title_h1 else "Judul tidak ditemukan"
        else:
            title_h1 = main_content.find('h1')
            metadata['judul'] = clean_text(title_h1.get_text()) if title_h1 else "Judul tidak ditemukan"
        date_element = main_content.find('div', class_='date date-item__headline')
        if date_element:
            date_text = clean_text(date_element.get_text())
            date_patterns = [
                r'(\d{1,2}\s+\w+\s+\d{4})\s+(\d{1,2}:\d{2})',
                r'(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})',
                r'(\d{1,2}\s+\w+\s+\d{4})',
            ]
            for pattern in date_patterns:
                match = re.search(pattern, date_text)
                if match:
                    if len(match.groups()) == 2:
                        metadata['waktu_terbit'] = f"{match.group(1)} {match.group(2)} WIB"
                    else:
                        metadata['waktu_terbit'] = f"{match.group(1)}"
                    break
            else:
                metadata['waktu_terbit'] = date_text
        else:
            metadata['waktu_terbit'] = "Waktu tidak ditemukan"
        editor_div = main_content.find('div', class_=lambda x: x == '' or x is None)
        if editor_div:
            editor_text = clean_text(editor_div.get_text())
            editor_patterns = [
                r'Red\s*:\s*([^<]+)',
                r'Editor\s*:\s*([^<]+)',
                r'Reporter\s*:\s*([^<]+)'
            ]
            for pattern in editor_patterns:
                match = re.search(pattern, editor_text)
                if match:
                    metadata['editor'] = clean_text(match.group(1))
                    break
            if not metadata['editor']:
                editor_link = editor_div.find('a')
                if editor_link:
                    metadata['editor'] = clean_text(editor_link.get_text())
        if not metadata['editor']:
            all_text = main_content.get_text()
            editor_match = re.search(r'Red\s*:\s*([^\n<]+)', all_text)
            if editor_match:
                metadata['editor'] = clean_text(editor_match.group(1))
            else:
                metadata['editor'] = "Editor tidak ditemukan"
        article_content = main_content.find('div', class_='article-content')
        if article_content:
            konten_artikel = extract_text_from_element(article_content)
        else:
            fallback_selectors = [
                '.article-content',
                '.article-body',
                '.content',
                '.post-content',
                '[itemprop="articleBody"]',
                '.detail-text'
            ]
            konten_artikel = ""
            for selector in fallback_selectors:
                content_elem = main_content.select_one(selector)
                if content_elem:
                    konten_artikel = extract_text_from_element(content_elem)
                    break
            if not konten_artikel:
                konten_artikel = extract_text_from_element(main_content)
        metadata['konten'] = konten_artikel
        metadata['panjang_konten'] = len(konten_artikel)
        return metadata, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# ==================== FUNGSI SCRAPING PENCARIAN ====================
def generate_search_id(keyword, startdate, enddate):
    """Generate unique search_id based on inputs (without page)"""
    input_str = f"{keyword}_{startdate}_{enddate}"
    return hashlib.md5(input_str.encode()).hexdigest()[:16]

def scrape_republika_search(keyword, startdate, enddate):
    """
    Scrape all pages from Republika.co.id search until no more results
    """
    all_results = []
    page = 1
    max_pages = 50  # Safety limit to prevent infinite loop
    status_msgs = []
    
    while page <= max_pages:
        try:
            q = quote_plus(keyword)
            url = f"https://republika.co.id/search/v3/all/{page}/?q={q}&latest_date=custom&startdate={startdate}&enddate={enddate}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            
            st.write(f"🔍 Scraping page {page}: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            selector = "#search > div.main-wrapper > main > div.main-content > div.container > div.results-section"
            results_section = soup.select_one(selector)
            
            if not results_section:
                fallback_selectors = [
                    'div.results-section',
                    '.results-section',
                    'main div.container div[class*="result"]',
                    '.search-results'
                ]
                for sel in fallback_selectors:
                    results_section = soup.select_one(sel)
                    if results_section:
                        st.write(f"✅ Found results with fallback: {sel}")
                        break
            
            if not results_section:
                status_msgs.append(f"❌ Results section not found on page {page}. Stopping.")
                break
            
            items = []
            item_selectors = [
                'div[class*="card"] a',
                'article a',
                '.search-item',
                '.result-item',
                'div.max-card'
            ]
            
            for sel in item_selectors:
                items = results_section.select(sel)
                if items:
                    st.write(f"✅ Found {len(items)} items on page {page} with selector: {sel}")
                    break
            else:
                items = results_section.find_all('a', href=re.compile(r'/berita/|/reads/'))
            
            if not items:
                status_msgs.append(f"✅ No more results on page {page}. Stopping.")
                break
            
            page_results = []
            for item in items:
                title_elem = item.find(['h1', 'h2', 'h3', 'h4']) or item
                title = title_elem.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                
                date_elem = item.find(class_=re.compile(r'date|time')) or item.find('span')
                date_text = date_elem.get_text(strip=True) if date_elem else ""
                date_match = re.search(r'(\d{1,2}\s+\w+\s+\d{4},\s+\d{1,2}:\d{2})', date_text)
                date = date_match.group(1) if date_match else "Date not found"
                
                href = item.get('href', '')
                if href.startswith('/'):
                    full_url = urljoin("https://republika.co.id", href)
                else:
                    full_url = href
                
                page_results.append({
                    'title': title[:200],
                    'date': date,
                    'url': full_url
                })
            
            all_results.extend(page_results)
            status_msgs.append(f"✅ Found {len(page_results)} results on page {page}")
            
            # Check for next page (look for pagination)
            next_page = soup.find('a', class_='next') or soup.find('a', text=re.compile(r'Next|Selanjutnya'))
            if not next_page:
                status_msgs.append("✅ No next page found. Stopping.")
                break
            
            page += 1
            time.sleep(2)  # Delay to avoid rate limiting
            
        except Exception as e:
            status_msgs.append(f"❌ Error on page {page}: {str(e)}. Stopping.")
            break
    
    return all_results, "\n".join(status_msgs)

# ==================== FUNGSI UNTUK APPEND KE CSV ====================
def append_to_csv(df, filename):
    """Append DataFrame to existing CSV file or create new one"""
    if os.path.exists(filename):
        # File exists, append without header
        df.to_csv(filename, mode='a', header=False, index=False)
        st.success(f"✅ Data appended to {filename}")
    else:
        # File doesn't exist, create new with header
        df.to_csv(filename, index=False)
        st.success(f"✅ New file created: {filename}")

# ==================== FUNGSI DOWNLOAD CSV ====================
def get_csv_download_link(df, filename, text):
    """Generate download link for CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ==================== FUNGSI UNTUK TAB TINJAUAN DATA ====================
def load_and_filter_data(start_date, end_date):
    """Memuat dan memfilter data berdasarkan tanggal"""
    csv_dir = "scrapper_result"
    
    # Load data
    try:
        keyword_df = pd.read_csv(os.path.join(csv_dir, "keyword_search.csv"))
    except:
        keyword_df = pd.DataFrame()
    
    try:
        results_df = pd.read_csv(os.path.join(csv_dir, "search_results.csv"))
    except:
        results_df = pd.DataFrame()
    
    try:
        metadata_df = pd.read_csv(os.path.join(csv_dir, "article_metadata.csv"))
        
        # Filter metadata berdasarkan tanggal jika tersedia
        if not metadata_df.empty and 'waktu_terbit' in metadata_df.columns:
            # Konversi kolom waktu_terbit ke datetime (dengan handling berbagai format)
            metadata_df['waktu_terbit_clean'] = pd.to_datetime(
                metadata_df['waktu_terbit'], 
                errors='coerce',
                format='mixed'
            )
            
            # Filter berdasarkan tanggal
            if start_date and end_date:
                mask = (metadata_df['waktu_terbit_clean'] >= pd.to_datetime(start_date)) & \
                       (metadata_df['waktu_terbit_clean'] <= pd.to_datetime(end_date))
                metadata_df = metadata_df[mask]
    
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        metadata_df = pd.DataFrame()
    
    return keyword_df, results_df, metadata_df

# ==================== PROSES UTAMA SCRAPING ====================
def process_republika_search(keyword, startdate_str, enddate_str):
    if not keyword.strip():
        st.error("❌ Masukkan keyword pencarian!")
        return None, None, None, None
    
    startdate = startdate_str or '2025-10-01'
    enddate = enddate_str or '2025-10-31'
    
    # Validasi format tanggal
    try:
        if startdate:
            datetime.strptime(startdate, '%Y-%m-%d')
        if enddate:
            datetime.strptime(enddate, '%Y-%m-%d')
    except ValueError:
        st.error("❌ Format tanggal harus YYYY-MM-DD!")
        return None, None, None, None
    
    with st.spinner("🔄 Sedang melakukan scraping..."):
        results_list, status = scrape_republika_search(keyword, startdate, enddate)
    
    if not results_list:
        st.warning("❌ Tidak ada hasil yang ditemukan!")
        return None, None, None, None
    
    search_id = generate_search_id(keyword, startdate, enddate)
    timestamp_search = datetime.now().isoformat()
    num_results = len(results_list)
    results_json = json.dumps(results_list, ensure_ascii=False)
    
    df_keyword_search = pd.DataFrame([{
        'search_id': search_id,
        'keyword': keyword,
        'source_type': 'Republika Search',
        'num_results': num_results,
        'results': results_json,
        'timestamp_search': timestamp_search
    }])
    
    df_results = pd.DataFrame(results_list)
    
    # Scraping metadata artikel untuk setiap URL
    articles_metadata = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, result in enumerate(results_list):
        url = result['url']
        status_text.text(f"📄 Scraping artikel {i+1}/{len(results_list)}: {url}")
        metadata, error = extract_republika_article(url)
        if metadata:
            metadata['search_id'] = search_id
            metadata['article_id'] = hashlib.md5(url.encode()).hexdigest()[:16]
            metadata['timestamp_ekstraksi'] = datetime.now().isoformat()
            articles_metadata.append(metadata)
            time.sleep(2)  # Delay
        else:
            st.warning(f"⚠️ Failed to scrape {url}: {error}")
        
        progress_bar.progress((i + 1) / len(results_list))
    
    status_text.empty()
    progress_bar.empty()
    
    df_metadata = pd.DataFrame(articles_metadata)
    
    # Simpan ke CSV dengan nama file spesifik (seperti sebelumnya)
    csv_dir = "scrapper_result"
    os.makedirs(csv_dir, exist_ok=True)
    
    keyword_csv_path = os.path.join(csv_dir, f"keyword_search_{search_id}.csv")
    results_csv_path = os.path.join(csv_dir, f"search_results_{search_id}.csv")
    metadata_csv_path = os.path.join(csv_dir, f"article_metadata_{search_id}.csv")
    
    df_keyword_search.to_csv(keyword_csv_path, index=False)
    df_results.to_csv(results_csv_path, index=False)
    df_metadata.to_csv(metadata_csv_path, index=False)
    
    # TAMBAHAN: Append ke file CSV utama
    main_keyword_csv = os.path.join(csv_dir, "keyword_search.csv")
    main_results_csv = os.path.join(csv_dir, "search_results.csv")
    main_metadata_csv = os.path.join(csv_dir, "article_metadata.csv")
    
    # Append data ke file utama
    append_to_csv(df_keyword_search, main_keyword_csv)
    append_to_csv(df_results, main_results_csv)
    if not df_metadata.empty:
        append_to_csv(df_metadata, main_metadata_csv)
    
    # Upload ke GitHub
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Upload file individual
        upload_to_github(
            f"{SCRAPPER_RESULT_PATH}/keyword_search_{search_id}.csv",
            df_keyword_search,
            f"Scraping keyword search {search_id} - {timestamp}"
        )
        
        upload_to_github(
            f"{SCRAPPER_RESULT_PATH}/search_results_{search_id}.csv",
            df_results,
            f"Scraping search results {search_id} - {timestamp}"
        )
        
        if not df_metadata.empty:
            upload_to_github(
                f"{SCRAPPER_RESULT_PATH}/article_metadata_{search_id}.csv",
                df_metadata,
                f"Scraping article metadata {search_id} - {timestamp}"
            )
        
        # Upload file utama (append)
        upload_to_github(
            f"{SCRAPPER_RESULT_PATH}/keyword_search.csv",
            df_keyword_search,
            f"Update keyword search utama - {timestamp}"
        )
        
        upload_to_github(
            f"{SCRAPPER_RESULT_PATH}/search_results.csv",
            df_results,
            f"Update search results utama - {timestamp}"
        )
        
        if not df_metadata.empty:
            upload_to_github(
                f"{SCRAPPER_RESULT_PATH}/article_metadata.csv",
                df_metadata,
                f"Update article metadata utama - {timestamp}"
            )
        
        st.success("✅ Data berhasil diupload ke GitHub")
        
    except Exception as e:
        st.error(f"❌ Gagal upload ke GitHub: {str(e)}")
    
    return df_results, df_keyword_search, df_metadata, search_id, num_results, len(articles_metadata), status

# ==================== INTERFACE STREAMLIT DENGAN TAB ====================
def main():
    # Header
    st.title("🔍 Republika.co.id Search Scraper & Analisis Sentimen")
    
    # Buat tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Scraping", 
        "📈 Tinjauan Data", 
        "🧠 Analisis Sentimen", 
        "📋 Dashboard Sentimen"
    ])
    
    # ==================== TAB 1: SCRAPING ====================
    with tab1:
        st.header("📊 Scraping Data Republika.co.id")
        st.markdown("""
        **Ekstrak hasil pencarian berdasarkan keyword ke dalam skema KeywordSearchResult & ArticleMetadata**
        
        **Fitur:**
        - ✅ Scraping semua halaman otomatis
        - ✅ Ekstraksi otomatis title, date, URL
        - ✅ Scraping metadata artikel untuk setiap hasil
        - ✅ Generate Search ID unik
        - ✅ Simpan ke CSV otomatis
        - ✅ **DATA DITAMBAHKAN KE FILE CSV UTAMA** (keyword_search.csv, search_results.csv, article_metadata.csv)
        - ✅ **SYNC OTOMATIS KE GITHUB** (https://github.com/abdfajar/republika_sentiner)
        """)
        
        # Sidebar untuk input (hanya di tab scraping)
        with st.sidebar:
            st.header("🔎 Parameter Pencarian")
            keyword_input = st.text_input(
                "🔑 Keyword Pencarian",
                value="MBG",
                placeholder="e.g., MBG"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                startdate_input = st.text_input(
                    "📅 Tanggal Mulai",
                    value="2025-10-01",
                    placeholder="YYYY-MM-DD"
                )
            with col2:
                enddate_input = st.text_input(
                    "📅 Tanggal Selesai",
                    value="2025-10-31",
                    placeholder="YYYY-MM-DD"
                )
            
            search_btn = st.button(
                "🚀 Cari & Scrap Semua Halaman",
                type="primary",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("### GitHub Operations")
            
            col_sync1, col_sync2 = st.columns(2)
            with col_sync1:
                if st.button("📤 Sync ke GitHub", use_container_width=True):
                    with st.spinner("Menyinkronisasi ke GitHub..."):
                        success_count, total_count = sync_to_github()
                        if success_count > 0:
                            st.success(f"✅ {success_count}/{total_count} file berhasil disinkronisasi ke GitHub")
                        else:
                            st.warning("Tidak ada file yang berhasil disinkronisasi")
            
            with col_sync2:
                if st.button("📥 Load dari GitHub", use_container_width=True):
                    with st.spinner("Memuat data dari GitHub..."):
                        success_count, total_count = load_from_github()
                        if success_count > 0:
                            st.success(f"✅ {success_count}/{total_count} file berhasil dimuat dari GitHub")
                        else:
                            st.warning("Tidak ada file yang berhasil dimuat dari GitHub")
            
            st.markdown("---")
            st.markdown("### Contoh Pencarian")
            if st.button("MBG (Oktober 2025)", use_container_width=True):
                st.session_state.keyword = "MBG"
                st.session_state.startdate = "2025-10-01"
                st.session_state.enddate = "2025-10-31"
            
            if st.button("Prabowo (September 2025)", use_container_width=True):
                st.session_state.keyword = "Prabowo"
                st.session_state.startdate = "2025-09-01"
                st.session_state.enddate = "2025-09-30"
        
        # Set nilai dari session state jika ada
        if 'keyword' in st.session_state:
            keyword_input = st.session_state.keyword
        if 'startdate' in st.session_state:
            startdate_input = st.session_state.startdate
        if 'enddate' in st.session_state:
            enddate_input = st.session_state.enddate
        
        # Proses pencarian ketika tombol ditekan
        if search_btn:
            df_results, df_keyword_search, df_metadata, search_id, num_results, num_articles, status = process_republika_search(
                keyword_input, startdate_input, enddate_input
            )
            
            if df_results is not None:
                # Tampilkan hasil
                st.success("✅ Scraping selesai!")
                
                # Summary hasil
                st.subheader("📊 Summary Hasil Pencarian")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔑 Keyword", keyword_input)
                with col2:
                    st.metric("📅 Periode", f"{startdate_input} s.d. {enddate_input}")
                with col3:
                    st.metric("📚 Total Hasil", num_results)
                with col4:
                    st.metric("📑 Metadata Artikel", num_articles)
                
                st.metric("💾 Search ID", search_id)
                
                # Tampilkan status scraping
                with st.expander("📋 Detail Status Scraping"):
                    st.text(status)
                
                # Tabel hasil pencarian
                st.subheader("📋 Tabel Hasil Pencarian")
                st.dataframe(df_results, use_container_width=True)
                
                # Tabel skema keyword search
                st.subheader("💾 Skema KeywordSearchResult")
                st.dataframe(df_keyword_search, use_container_width=True)
                
                # Tabel metadata artikel
                if not df_metadata.empty:
                    st.subheader("📚 Tabel Metadata Artikel")
                    st.dataframe(df_metadata, use_container_width=True)
                
                # Download section
                st.subheader("💾 Download CSV Files")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(get_csv_download_link(df_keyword_search, "keyword_search.csv", "📥 Download Keyword Search CSV"), 
                               unsafe_allow_html=True)
                with col2:
                    st.markdown(get_csv_download_link(df_results, "search_results.csv", "📥 Download Search Results CSV"), 
                               unsafe_allow_html=True)
                with col3:
                    if not df_metadata.empty:
                        st.markdown(get_csv_download_link(df_metadata, "article_metadata.csv", "📥 Download Article Metadata CSV"), 
                                   unsafe_allow_html=True)
                
                # Informasi file tersimpan
                st.info(f"""
                **📁 File CSV Tersimpan:**
                - `scrapper_result/keyword_search_{search_id}.csv`
                - `scrapper_result/search_results_{search_id}.csv`
                - `scrapper_result/article_metadata_{search_id}.csv`
                
                **📥 Data juga ditambahkan ke file utama:**
                - `scrapper_result/keyword_search.csv`
                - `scrapper_result/search_results.csv`
                - `scrapper_result/article_metadata.csv`
                
                **🌐 Data telah diupload ke GitHub:**
                - `https://github.com/{GITHUB_REPO}/tree/main/{SCRAPPER_RESULT_PATH}`
                """)
    
    # ==================== TAB 2: TINJAUAN DATA ====================
    with tab2:
        st.header("📈 Tinjauan Data Hasil Scraping")
        st.markdown("Menampilkan data dari file CSV yang telah di-scrap")
        
        # Input tanggal untuk filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Tanggal Mulai", value=datetime(2025, 10, 1))
        with col2:
            end_date = st.date_input("Tanggal Selesai", value=datetime(2025, 10, 31))
        
        # Tombol untuk memuat data
        if st.button("🔍 Muat Data", key="load_data"):
            with st.spinner("Memuat data..."):
                keyword_df, results_df, metadata_df = load_and_filter_data(start_date, end_date)
                
                # Tampilkan keyword_search.csv
                st.subheader("📋 Keyword Search Data")
                if not keyword_df.empty:
                    st.dataframe(keyword_df, use_container_width=True)
                    st.metric("Jumlah Record", len(keyword_df))
                else:
                    st.warning("Data keyword_search.csv tidak ditemukan atau kosong")
                
                # Tampilkan search_results.csv
                st.subheader("📋 Search Results Data")
                if not results_df.empty:
                    st.dataframe(results_df, use_container_width=True)
                    st.metric("Jumlah Record", len(results_df))
                else:
                    st.warning("Data search_results.csv tidak ditemukan atau kosong")
                
                # Tampilkan article_metadata.csv
                st.subheader("📋 Article Metadata Data")
                if not metadata_df.empty:
                    st.dataframe(metadata_df, use_container_width=True)
                    st.metric("Jumlah Record", len(metadata_df))
                    
                    # Tampilkan statistik
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'panjang_konten' in metadata_df.columns:
                            st.metric("Rata-rata Panjang Konten", f"{metadata_df['panjang_konten'].mean():.0f} karakter")
                    with col2:
                        if 'editor' in metadata_df.columns:
                            unique_editors = metadata_df['editor'].nunique()
                            st.metric("Jumlah Editor Unik", unique_editors)
                    with col3:
                        if 'waktu_terbit' in metadata_df.columns:
                            st.metric("Rentang Tanggal", f"{start_date} to {end_date}")
                else:
                    st.warning("Data article_metadata.csv tidak ditemukan atau kosong")
    
    # ==================== TAB 3: ANALISIS SENTIMEN ====================
    with tab3:
        st.header("🧠 Analisis Sentimen")
        st.markdown("""
        Analisis sentimen pada artikel yang telah di-scrap menggunakan model AI:
        - **Sentiment Analysis**: IndoBERT model
        - **Named Entity Recognition**: BERT-base Indonesian NER
        - **Trigram Analysis**: Pola kata berurutan
        - **TF-IDF Analysis**: Kata kunci penting
        """)
        
        # Input tanggal untuk analisis
        col1, col2 = st.columns(2)
        with col1:
            analysis_start_date = st.date_input("Tanggal Mulai Analisis", value=datetime(2025, 10, 1), key="analysis_start")
        with col2:
            analysis_end_date = st.date_input("Tanggal Selesai Analisis", value=datetime(2025, 10, 31), key="analysis_end")
        
        # Peringatan tentang waktu loading model
        st.info("⚠️ **Note**: Loading model AI pertama kali mungkin memakan waktu beberapa menit.")
        
        # Tombol untuk analisis
        if st.button("🚀 Analisis Sentimen", type="primary"):
            with st.spinner("Melakukan analisis sentimen komprehensif..."):
                df_sentiment, df_ner, df_trigram, df_tfidf = analyze_sentiment_comprehensive(
                    analysis_start_date, analysis_end_date
                )
                
                # Tampilkan hasil Prediksi Sentimen
                st.subheader("📊 Prediksi Sentimen")
                if not df_sentiment.empty:
                    st.dataframe(df_sentiment, use_container_width=True)
                    
                    # Visualisasi distribusi sentimen
                    sentiment_counts = df_sentiment['sentimen'].value_counts()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.bar_chart(sentiment_counts)
                    
                    with col2:
                        # Pie chart menggunakan bar chart sebagai alternatif
                        st.write("**Distribusi Sentimen**")
                        for sentimen, count in sentiment_counts.items():
                            st.write(f"- {sentimen}: {count} artikel")
                    
                    # Statistik sentimen
                    total_articles = len(df_sentiment)
                    positive_pct = (len(df_sentiment[df_sentiment['sentimen'] == 'POSITIF']) / total_articles) * 100
                    negative_pct = (len(df_sentiment[df_sentiment['sentimen'] == 'NEGATIF']) / total_articles) * 100
                    neutral_pct = (len(df_sentiment[df_sentiment['sentimen'] == 'NETRAL']) / total_articles) * 100
                    
                    st.metric("Sentimen Positif", f"{positive_pct:.1f}%")
                    st.metric("Sentimen Negatif", f"{negative_pct:.1f}%")
                    st.metric("Sentimen Netral", f"{neutral_pct:.1f}%")
                    
                else:
                    st.warning("Tidak ada data prediksi sentimen")
                
                # Tampilkan hasil NER
                st.subheader("🏷️ Named Entity Recognition (NER)")
                if not df_ner.empty:
                    st.dataframe(df_ner, use_container_width=True)
                    st.bar_chart(df_ner.set_index('entity')['frekuensi'].head(10))
                else:
                    st.warning("Tidak ada data NER")
                
                # Tampilkan hasil Trigram
                st.subheader("🔤 Trigram Analysis")
                if not df_trigram.empty:
                    st.dataframe(df_trigram, use_container_width=True)
                    st.bar_chart(df_trigram.set_index('trigram')['frekuensi'].head(10))
                else:
                    st.warning("Tidak ada data trigram")
                
                # Tampilkan hasil TF-IDF
                st.subheader("📊 TF-IDF Analysis")
                if not df_tfidf.empty:
                    st.dataframe(df_tfidf, use_container_width=True)
                    st.bar_chart(df_tfidf.set_index('kata_kunci')['skor_tfidf'].head(10))
                else:
                    st.warning("Tidak ada data TF-IDF")
    
    # ==================== TAB 4: DASHBOARD SENTIMEN ====================
    with tab4:
        st.header("📋 Dashboard Sentimen Analysis Badan Gizi Nasional")
        st.markdown("Dashboard untuk memantau sentimen publik terhadap Badan Gizi Nasional")
        
        # Placeholder untuk dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Artikel", "156")
            st.metric("Sentimen Positif", "89")
        
        with col2:
            st.metric("Sentimen Negatif", "32")
            st.metric("Sentimen Netral", "35")
        
        with col3:
            st.metric("Tingkat Engagement", "78%")
            st.metric("Rata-rata Panjang Artikel", "1,245")
        
        # Visualisasi
        st.subheader("📈 Trend Sentimen Over Time")
        st.line_chart(pd.DataFrame({
            'Bulan': ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun'],
            'Positif': [12, 15, 18, 22, 25, 28],
            'Negatif': [8, 6, 5, 4, 3, 2],
            'Netral': [10, 12, 11, 13, 12, 11]
        }).set_index('Bulan'))
        
        st.subheader("🎯 Topik Populer")
        st.bar_chart(pd.DataFrame({
            'Topik': ['Program Stunting', 'Gizi Anak', 'Suplementasi', 'Edukasi Masyarakat', 'Kerjasama Internasional'],
            'Jumlah Artikel': [45, 38, 32, 28, 22]
        }).set_index('Topik'))
        
        st.subheader("📊 Distribusi Sentimen")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pie Chart Sentimen**")
            # Placeholder untuk pie chart
            sentiment_data = pd.DataFrame({
                'sentimen': ['Positif', 'Negatif', 'Netral'],
                'jumlah': [89, 32, 35]
            })
            st.bar_chart(sentiment_data.set_index('sentimen'))
        
        with col2:
            st.write("**Media Coverage**")
            media_data = pd.DataFrame({
                'media': ['Republika', 'Kompas', 'Detik', 'Tribun', 'Antara'],
                'jumlah': [45, 38, 32, 25, 16]
            })
            st.bar_chart(media_data.set_index('media'))
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"Dibuat dengan ❤️ menggunakan Streamlit | "
        f"Scraping data dari Republika.co.id | "
        f"GitHub: https://github.com/{GITHUB_REPO}"
    )

if __name__ == "__main__":
    main()
