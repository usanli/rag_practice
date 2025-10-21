# 📚 RAG Chat Uygulaması

OpenAI ve Pinecone kullanarak dokümanlar üzerinden soru-cevap yapabilen bir RAG (Retrieval Augmented Generation) uygulaması.

## 🌟 Özellikler

- 📄 **Çoklu Doküman Desteği**: PDF, TXT ve DOCX formatlarında doküman yükleme
- 🤖 **Premium Embedding**: OpenAI'ın text-embedding-3-large modeli ile yüksek kaliteli vektörize etme
- ☁️ **Cloud Depolama**: Pinecone vector database ile vektörleri bulutta saklama
- 💬 **Sohbet Arayüzü**: Streamlit tabanlı kullanıcı dostu chat interface
- 🧠 **Gelişmiş RAG Pipeline**: Dokümanlarınızdan ilgili bilgileri bulup GPT-4o ile akıllı analiz
- 📊 **İstatistikler**: Yüklenen doküman ve vector sayısı gösterimi
- 🎯 **Akıllı Filtreleme**: Similarity threshold ile kaliteli sonuçlar
- 🔍 **Kaynak Referansları**: Her cevap için detaylı kaynak bilgileri

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- OpenAI API key
- Pinecone API key

### Adımlar

1. **Repoyu klonlayın veya dosyaları indirin**

2. **Virtual environment oluşturun (opsiyonel ama önerilir)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Gerekli paketleri yükleyin**
```bash
pip install -r requirements.txt
```

4. **API anahtarlarınızı ayarlayın**

Proje klasöründe `.env` dosyası oluşturun ve aşağıdaki bilgileri girin:

```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Pinecone API Key
PINECONE_API_KEY=your-pinecone-api-key-here

# Pinecone Index Name (istediğiniz ismi verebilirsiniz)
PINECONE_INDEX_NAME=rag-documents
```

**API Key'leri Nereden Alınır?**

- **OpenAI API Key**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Pinecone API Key**: [https://app.pinecone.io/](https://app.pinecone.io/) (ücretsiz plan mevcut)

**Pinecone Index Name Nedir?**

Index adını siz belirlersiniz. Örneğin: `rag-documents`, `my-docs`, vb. 
Uygulama ilk çalıştığında bu isimde bir index yoksa otomatik olarak oluşturulacaktır.

## 📖 Kullanım

1. **Uygulamayı başlatın**
```bash
streamlit run app.py
```

2. **Tarayıcınızda açılacak olan uygulamada:**
   - Sol menüden dokümanlarınızı yükleyin (PDF, TXT veya DOCX)
   - "Dokümanları İşle ve Kaydet" butonuna tıklayın
   - Dokümanlarınız işlenip Pinecone'a kaydedilecek
   - Artık chat alanından sorularınızı sorabilirsiniz!

3. **Örnek sorular:**
   - "Bu dokümanda ana konular nelerdir?"
   - "X hakkında ne söyleniyor?"
   - "Y ile ilgili detayları özetle"

## 🏗️ Proje Yapısı

```
vectorize/
├── app.py                    # Ana Streamlit uygulaması
├── config.py                 # Yapılandırma ayarları
├── document_processor.py     # Doküman işleme modülü
├── vector_store.py          # Pinecone ve embedding yönetimi
├── requirements.txt         # Python bağımlılıkları
├── .env                     # API anahtarları (oluşturmanız gerekiyor)
└── README.md               # Bu dosya
```

## ⚙️ Yapılandırma

`config.py` dosyasından aşağıdaki ayarları özelleştirebilirsiniz:

- `EMBEDDING_MODEL`: Kullanılan embedding modeli (varsayılan: text-embedding-3-large)
- `CHAT_MODEL`: Chat için kullanılan model (varsayılan: gpt-4o)
- `EMBEDDING_DIMENSION`: Embedding boyutu (varsayılan: 3072)
- `CHUNK_SIZE`: Doküman chunk boyutu (varsayılan: 1200 karakter)
- `CHUNK_OVERLAP`: Chunk'lar arası örtüşme (varsayılan: 200 karakter)
- `TOP_K`: Query'de döndürülecek chunk sayısı (varsayılan: 8)
- `SIMILARITY_THRESHOLD`: Minimum benzerlik skoru (varsayılan: 0.25)
- `MAX_CONTEXT_LENGTH`: Maksimum context uzunluğu (varsayılan: 8000 karakter)

### Model Alternatifleri

#### Chat Modelleri:
- `CHAT_MODEL = "gpt-4o"` - Hızlı, güçlü, güvenilir (önerilen)
- `CHAT_MODEL = "gpt-4o-mini"` - Ekonomik seçenek
- `CHAT_MODEL = "gpt-5"` - En yeni model (API erişimi gerekebilir)
- `CHAT_MODEL = "gpt-5-mini"` - Ekonomik GPT-5

#### Embedding Modelleri:
- `EMBEDDING_MODEL = "text-embedding-3-large"` - En kaliteli (3072 dimension)
- `EMBEDDING_MODEL = "text-embedding-3-small"` - Ekonomik (1536 dimension)

**ÖNEMLİ:** Embedding modelini değiştirirseniz `EMBEDDING_DIMENSION` değerini de güncelleyin:
- text-embedding-3-large → 3072
- text-embedding-3-small → 1536

## 💡 Nasıl Çalışır?

1. **Doküman Yükleme**: Kullanıcı PDF/TXT/DOCX dosyalarını yükler
2. **Metin Çıkarma**: Dokümanlardan metin çıkarılır
3. **Chunking**: Metin küçük parçalara (chunks) bölünür
4. **Embedding**: Her chunk OpenAI API ile vektöre dönüştürülür
5. **Depolama**: Vektörler Pinecone cloud'da saklanır
6. **Sorgulama**: Kullanıcı soru sorduğunda:
   - Soru vektöre dönüştürülür
   - Pinecone'da benzer chunk'lar bulunur
   - Bulunan chunk'lar GPT'ye context olarak verilir
   - Model context'e göre cevap üretir

## 🔧 Sorun Giderme

### "API keys bulunamadı" hatası
- `.env` dosyasının proje klasöründe olduğundan emin olun
- API key'lerin doğru girildiğini kontrol edin

### "Pinecone bağlantı hatası"
- Pinecone environment bilgisinin doğru olduğundan emin olun
- İnternet bağlantınızı kontrol edin
- Pinecone hesabınızın aktif olduğundan emin olun

### "Doküman işlenemedi" hatası
- Dosya formatının desteklendiğinden emin olun (PDF, TXT, DOCX)
- Dosyanın bozuk olmadığını kontrol edin
- PDF'ler için: Dosyanın şifreli olmadığından emin olun

### "Vector dimension mismatch" hatası

**Sorun:** Mevcut Pinecone index'iniz farklı bir dimension'da oluşturulmuş.

**Çözüm 1 (Önerilen):** `.env` dosyasında index adını değiştirin:
```env
PINECONE_INDEX_NAME=rag-documents-v2
```
Yeni index otomatik oluşturulacak. Eski verileriniz kaybolmayacak (eski index'te kalacak).

**Çözüm 2:** Pinecone dashboard'dan ([https://app.pinecone.io/](https://app.pinecone.io/)) eski index'i silin ve uygulamayı yeniden başlatın.

**Doğrulama:** Terminalde şu komutu çalıştırın:
```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Index Name:', os.getenv('PINECONE_INDEX_NAME'))"
```

### "gpt-5" model hatası

Eğer GPT-5'e erişiminiz yoksa, `config.py` dosyasında:
```python
CHAT_MODEL = "gpt-4o"  # Tüm hesaplarda çalışır
```

## 📊 Maliyet Bilgileri

### Premium Yapılandırma (Varsayılan):
- **Embedding**: text-embedding-3-large - $0.13 / 1M tokens
- **Chat**: gpt-4o - Input: $5 / 1M tokens, Output: $15 / 1M tokens

### Ekonomik Yapılandırma:
```python
# config.py
EMBEDDING_MODEL = "text-embedding-3-small"  # $0.02 / 1M tokens
EMBEDDING_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"  # Input: $0.15, Output: $0.60 / 1M tokens
CHUNK_SIZE = 800
TOP_K = 5
```

## 💡 İpuçları

### Daha İyi Sonuçlar İçin:
1. **Spesifik Sorular Sorun**: "X hakkında ne söyleniyor?" yerine "X'in Y üzerindeki etkisi nedir?"
2. **Context Verin**: Sorunuza context ekleyin
3. **Chunk Sayısını Ayarlayın**: `config.py`'da `TOP_K` değerini artırın (daha fazla context)
4. **Similarity Threshold'u Ayarlayın**: Daha az ama daha ilgili sonuçlar için artırın (ör: 0.5)

### GPT-5 Kullanımı:

GPT-5 modellerini kullanmak isterseniz `config.py` dosyasında:
```python
CHAT_MODEL = "gpt-5"  # veya "gpt-5-mini"
```

**Not:** GPT-5 modelleri temperature parametresini desteklemez (varsayılan 1 kullanır).

## 📝 Notlar

- İlk çalıştırmada Pinecone index'i otomatik olarak oluşturulur
- Her doküman işleme ve sorgulama OpenAI API kullandığı için ücrete tabidir
- Pinecone ücretsiz planı 100.000 vektöre kadar destekler
- Maliyet önceliğinizse ekonomik yapılandırmayı kullanın

## 📄 Lisans

Bu proje eğitim amaçlı oluşturulmuştur.

## 🤝 Katkıda Bulunma

Önerileriniz ve katkılarınız için pull request açabilirsiniz!
