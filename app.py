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

# ==================== PROSES UTAMA ====================
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
    
    return df_results, df_keyword_search, df_metadata, search_id, num_results, len(articles_metadata), status

# ==================== INTERFACE STREAMLIT ====================
def main():
    # Header
    st.title("🔍 Republika.co.id Search Scraper")
    st.markdown("""
    **Ekstrak hasil pencarian berdasarkan keyword ke dalam skema KeywordSearchResult & ArticleMetadata**
    
    **Fitur:**
    - ✅ Scraping semua halaman otomatis
    - ✅ Ekstraksi otomatis title, date, URL
    - ✅ Scraping metadata artikel untuk setiap hasil
    - ✅ Generate Search ID unik
    - ✅ Simpan ke CSV otomatis
    - ✅ **DATA DITAMBAHKAN KE FILE CSV UTAMA** (keyword_search.csv, search_results.csv, article_metadata.csv)
    """)
    
    # Sidebar untuk input
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
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Dibuat dengan ❤️ menggunakan Streamlit | "
        "Scraping data dari Republika.co.id"
    )

if __name__ == "__main__":
    main()
