"""
RAG Chat Application with Streamlit
Main application file for document upload and chat interface
"""

import streamlit as st
import os
from dotenv import load_dotenv
from document_processor import process_document
from vector_store import VectorStore
from openai import OpenAI
import config

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chat Application",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = 0
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None


def initialize_clients():
    """Initialize OpenAI and Pinecone clients"""
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'rag-documents')
        
        if not all([openai_api_key, pinecone_api_key]):
            st.error("❌ API keys bulunamadı! Lütfen .env dosyasını kontrol edin.")
            st.info("📝 .env dosyasında şu değişkenler olmalı: OPENAI_API_KEY, PINECONE_API_KEY")
            return False
        
        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore(
                api_key=pinecone_api_key,
                index_name=pinecone_index_name,
                openai_api_key=openai_api_key
            )
        
        # Initialize OpenAI client
        if st.session_state.openai_client is None:
            st.session_state.openai_client = OpenAI(api_key=openai_api_key)
        
        return True
    except Exception as e:
        st.error(f"❌ Bağlantı hatası: {str(e)}")
        return False


def generate_rag_response(query: str) -> str:
    """
    Generate response using enhanced RAG pipeline with reasoning
    Optimized for o1-preview model's deep analytical capabilities
    
    Args:
        query: User's question
        
    Returns:
        AI-generated response with reasoning
    """
    try:
        # Query vector store for relevant chunks
        relevant_chunks = st.session_state.vector_store.query_vectors(
            query_text=query,
            top_k=config.TOP_K
        )
        
        if not relevant_chunks:
            return "Üzgünüm, yüklediğiniz dokümanlarda bu soruyla ilgili bilgi bulamadım. Lütfen önce doküman yüklediğinizden emin olun."
        
        # Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in relevant_chunks 
            if chunk['score'] >= config.SIMILARITY_THRESHOLD
        ]
        
        if not filtered_chunks:
            # Threshold çok yüksekse, en iyi sonuçları kullan
            if relevant_chunks:
                st.warning(f"⚠️ Düşük benzerlik (en yüksek: {relevant_chunks[0]['score']:.2%}). En iyi {len(relevant_chunks[:3])} sonuç kullanılıyor.")
                filtered_chunks = relevant_chunks[:3]  # En az 3 chunk kullan
            else:
                return "Dokümanlarda bu soruyla ilgili bilgi bulunamadı. Lütfen sorunuzu farklı şekilde sormayı deneyin."
        
        # Build enriched context with structure
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(filtered_chunks):
            context_parts.append(
                f"=== KAYNAK {i+1} ===\n"
                f"Dosya: {chunk['filename']}\n"
                f"İlgililik Skoru: {chunk['score']:.3f}\n"
                f"İçerik:\n{chunk['text']}\n"
            )
            sources.append({
                'filename': chunk['filename'],
                'score': chunk['score']
            })
        
        context = "\n\n".join(context_parts)
        
        # Limit context but preserve important info
        if len(context) > config.MAX_CONTEXT_LENGTH:
            context = context[:config.MAX_CONTEXT_LENGTH] + "\n\n[... içerik uzunluk nedeniyle kısaltıldı ...]"
        
        # Enhanced prompt for GPT-5 reasoning
        system_prompt = """Sen bir uzman RAG (Retrieval Augmented Generation) asistanısın ve İnsan Kaynakları/CV analizi konusunda uzmansın. 

GÖREV:
1. Verilen doküman içeriğini DİKKATLİCE ve DETAYLI analiz et
2. Kullanıcının sorusuyla İLGİLİ TÜM bilgileri belirle ve BİRLEŞTİR
3. Birden fazla dokümandan gelen bilgileri SENTEZLE
4. Kişi isimleri, pozisyonlar, beceriler, deneyimler gibi detayları DİKKATLİCE NOT ET
5. Sayısal sorularda (kaç kişi, kaç yıl, vb.) DİKKATLİ SAY ve doğru rakam ver
6. Liste soruları için KAPSAMLI listeler oluştur
7. Verilen kaynaklara dayalı mantıksal ÇIKARIMLAR yap
8. Yanıtlarını Türkçe, net, yapılandırılmış ve profesyonel şekilde yaz

CEVAP FORMATI:
- Net ve direkt cevap ver
- Örnekler ve isimler kullan
- Listelerde madde işareti kullan
- Sayıları ve istatistikleri vurgula

Senin gücün: CV analizi, kişi eşleştirme, beceri değerlendirme, derin analiz ve kapsamlı sentez."""

        user_prompt = f"""Aşağıda {len(filtered_chunks)} farklı doküman içeriği var. Bunları analiz ederek soruyu yanıtla.

=== DOKÜMAN İÇERİKLERİ ===
{context}

=== KULLANICI SORUSU ===
{query}

=== TALİMATLAR ===
- TÜM dokümanlardaki ilgili bilgileri BİRLEŞTİR
- Sayısal sorularda (kaç kişi, kaç tane vb.) DİKKAT ET ve doğru say
- İsimleri, pozisyonları, becerileri belirgin şekilde BELIRT
- Liste soruları için KAPSAMLI listeler oluştur
- Her bilgi için KAYNAK dosyayı referans ver
- Net, yapılandırılmış ve DETAYLI cevap ver

ŞİMDİ YANIT VER:"""
        
        # Generate response using OpenAI GPT-5
        # GPT-5 has specific API requirements
        try:
            # GPT-5 does not support custom temperature, only default (1)
            if config.CHAT_MODEL.startswith("gpt-5"):
                response = st.session_state.openai_client.chat.completions.create(
                    model=config.CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=config.MAX_COMPLETION_TOKENS
                    # temperature not supported for GPT-5
                )
            else:
                # Other models support temperature
                response = st.session_state.openai_client.chat.completions.create(
                    model=config.CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config.TEMPERATURE,
                    max_completion_tokens=config.MAX_COMPLETION_TOKENS
                )
        except Exception as api_error:
            error_str = str(api_error)
            # Fallback for max_tokens parameter
            if "max_completion_tokens" in error_str:
                response = st.session_state.openai_client.chat.completions.create(
                    model=config.CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=config.TEMPERATURE if not config.CHAT_MODEL.startswith("gpt-5") else 1,
                    max_tokens=config.MAX_COMPLETION_TOKENS
                )
            else:
                raise
        
        answer = response.choices[0].message.content
        
        # Debug: Check if answer is empty
        if not answer or answer.strip() == "":
            answer = "⚠️ Model cevap üretemedi. Lütfen sorunuzu daha detaylı sorun veya farklı şekilde ifade edin."
        
        # Add enhanced sources section with unique filenames
        # Group by filename and keep the highest score
        unique_sources = {}
        for source in sources:
            filename = source['filename']
            score = source['score']
            if filename not in unique_sources or score > unique_sources[filename]:
                unique_sources[filename] = score
        
        # Sort by score (descending)
        sorted_sources = sorted(unique_sources.items(), key=lambda x: x[1], reverse=True)
        
        sources_text = "\n\n---\n**📚 Kullanılan Kaynaklar:**\n"
        for i, (filename, score) in enumerate(sorted_sources[:5], 1):
            sources_text += f"{i}. {filename} (İlgililik: {score:.1%})\n"
        
        # Make sure answer is shown before sources
        full_response = f"{answer}\n{sources_text}"
        
        return full_response
        
    except Exception as e:
        error_msg = str(e)
        
        # Log the full error for debugging
        import traceback
        full_error = traceback.format_exc()
        
        if "model_not_found" in error_msg or "does not exist" in error_msg:
            return f"""⚠️ **GPT-5 Model Bulunamadı!**

API key'iniz GPT-5'e erişimi desteklemiyor olabilir.

**ÇÖZÜM:** `config.py` dosyasında modeli değiştirin:
```python
CHAT_MODEL = "gpt-4o"  # Bu kesin çalışır
```

Sonra Streamlit'i yeniden başlatın (Ctrl+C, sonra `streamlit run app.py`)

Hata detayı: {error_msg}"""
        
        return f"""❌ **Cevap Üretme Hatası**

{error_msg}

**Debug Bilgisi:**
- Model: {config.CHAT_MODEL}
- Context uzunluğu: {len(context) if 'context' in locals() else 'N/A'} karakter
- Chunk sayısı: {len(filtered_chunks) if 'filtered_chunks' in locals() else 'N/A'}

Lütfen terminal çıktısını kontrol edin veya farklı bir model deneyin."""


# Main UI
st.title("📚 RAG Chat Uygulaması")
st.markdown("Dokümanlarınızı yükleyin ve sorularınızı sorun!")

# Initialize clients
if not initialize_clients():
    st.stop()

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Doküman Yönetimi")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Dokümanları yükleyin (PDF, TXT, DOCX)",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    # Process button
    if uploaded_files:
        st.info(f"📋 {len(uploaded_files)} dosya seçildi")
        
        if st.button("🚀 Dokümanları İşle ve Kaydet", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_chunks = 0
            
            for idx, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"İşleniyor: {file.name}")
                    
                    # Reset file pointer
                    file.seek(0)
                    
                    # Process document
                    chunks = process_document(file, file.name)
                    
                    if not chunks:
                        st.warning(f"⚠️ {file.name}: Boş doküman, atlanıyor")
                        continue
                    
                    # Embed and store
                    num_chunks = st.session_state.vector_store.embed_and_store(chunks)
                    total_chunks += num_chunks
                    
                    st.success(f"✅ {file.name}: {num_chunks} chunk işlendi")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                except Exception as e:
                    error_msg = str(e)
                    if "dimension" in error_msg.lower() and "1536" in error_msg:
                        st.error(f"""❌ **DIMENSION HATASI!**
                        
Pinecone index'iniz eski dimension'da (1536). Yeni embedding modeli 3072 kullanıyor.

**ÇÖZÜM:** .env dosyanızda index adını değiştirin:
```
PINECONE_INDEX_NAME=rag-documents-v2
```
Sonra sayfayı yenileyin (F5) ve tekrar deneyin.""")
                        break
                    else:
                        st.error(f"❌ {file.name} işlenirken hata: {str(e)}")
            
            status_text.empty()
            progress_bar.empty()
            
            st.session_state.documents_processed += len(uploaded_files)
            st.success(f"✅ {len(uploaded_files)} doküman başarıyla işlendi! ({total_chunks} chunk)")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 İstatistikler")
    try:
        stats = st.session_state.vector_store.get_index_stats()
        st.metric("Toplam Vector", stats['total_vectors'])
        st.metric("İşlenen Doküman", st.session_state.documents_processed)
    except:
        st.info("İstatistikler yükleniyor...")
    
    st.divider()
    
    # Clear data button
    if st.button("🗑️ Tüm Verileri Temizle", type="secondary"):
        if st.session_state.vector_store:
            st.session_state.vector_store.delete_all_vectors()
            st.session_state.chat_history = []
            st.session_state.documents_processed = 0
            st.success("✅ Tüm veriler temizlendi!")
            st.rerun()

# Main chat area
st.header("💬 Soru Sorun")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Dokümanlardaki bilgiler hakkında bir soru sorun..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            response = generate_rag_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Instructions if no documents
if st.session_state.documents_processed == 0:
    st.info("""
    👋 **Hoş geldiniz!**
    
    Başlamak için:
    1. Sol taraftaki menüden dokümanlarınızı yükleyin (PDF, TXT, DOCX)
    2. "Dokümanları İşle ve Kaydet" butonuna tıklayın
    3. Dokümanlarınız hakkında sorular sormaya başlayın!
    """)

