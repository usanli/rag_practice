"""
Otomatik .env dosyası düzeltme scripti
Pinecone index adını günceller
"""

import os
from pathlib import Path

def fix_env_file():
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env dosyası bulunamadı!")
        print("📝 Lütfen .env dosyası oluşturun:")
        print("""
OPENAI_API_KEY=your-key-here
PINECONE_API_KEY=your-key-here
PINECONE_INDEX_NAME=rag-documents-v2
        """)
        return False
    
    # .env dosyasını oku
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Index adını güncelle
    updated = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('PINECONE_INDEX_NAME='):
            old_value = line.strip()
            new_line = 'PINECONE_INDEX_NAME=rag-documents-v2\n'
            new_lines.append(new_line)
            updated = True
            print(f"🔄 Değiştirildi:")
            print(f"   Eski: {old_value}")
            print(f"   Yeni: PINECONE_INDEX_NAME=rag-documents-v2")
        else:
            new_lines.append(line)
    
    # Eğer PINECONE_INDEX_NAME yoksa ekle
    if not updated:
        if not any('PINECONE_INDEX_NAME' in line for line in new_lines):
            new_lines.append('\n# Pinecone Index Name\n')
            new_lines.append('PINECONE_INDEX_NAME=rag-documents-v2\n')
            print("➕ PINECONE_INDEX_NAME eklendi: rag-documents-v2")
            updated = True
    
    if updated:
        # Backup oluştur
        backup_path = Path(".env.backup")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"💾 Yedek oluşturuldu: {backup_path}")
        
        # Yeni .env dosyasını yaz
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print("✅ .env dosyası güncellendi!")
        print("\n📋 Şimdi yapmanız gerekenler:")
        print("1. Streamlit'i yeniden başlatın (Ctrl+C sonra 'streamlit run app.py')")
        print("2. Tarayıcıda sayfayı yenileyin (F5)")
        print("3. Dokümanlarınızı tekrar yükleyin")
        return True
    else:
        print("ℹ️ Değişiklik gerekmedi.")
        return False

if __name__ == "__main__":
    print("🔧 Pinecone Index Adı Güncelleniyor...\n")
    fix_env_file()

