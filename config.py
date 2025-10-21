"""
Configuration settings for the RAG application
Optimized for GPT-5 - Best for reasoning, coding, and agentic tasks
"""

# OpenAI Settings - Premium Models
EMBEDDING_MODEL = "text-embedding-3-large"  # Yüksek kaliteli embeddings (3072 dimension)

# Chat Model Options:
# - "gpt-4o" (ÖNERİLEN) - Hızlı, güçlü, güvenilir, tüm hesaplarda çalışır
# - "gpt-5" - En yeni model (API erişimi gerekebilir)
# - "gpt-5-mini" - Ekonomik GPT-5
# - "gpt-4o-mini" - Ekonomik GPT-4
CHAT_MODEL = "gpt-4o"  # GPT-4o - Hızlı, güvenilir ve güçlü

EMBEDDING_DIMENSION = 3072  # text-embedding-3-large için

# Document Processing Settings - Optimized
CHUNK_SIZE = 1200  # Daha büyük chunk'lar = daha fazla context
CHUNK_OVERLAP = 200  # Daha fazla overlap = bilgi kaybı azalır

# Pinecone Settings - Enhanced Retrieval
TOP_K = 8  # Daha fazla chunk = daha zengin context
SIMILARITY_THRESHOLD = 0.25  # Minimum similarity score (0.25 = daha esnek)

# Chat Settings - Optimized for GPT-5
MAX_CONTEXT_LENGTH = 8000  # GPT-5 için geniş context window
TEMPERATURE = 0.3  # Diğer modeller için (GPT-5 varsayılan 1 kullanır)
MAX_COMPLETION_TOKENS = 2000  # Maksimum yanıt uzunluğu (GPT-5 için max_completion_tokens)

# NOT: GPT-5 modelleri temperature parametresini desteklemez (varsayılan 1 kullanır)

