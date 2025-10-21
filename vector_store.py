"""
Vector store module for interacting with Pinecone and OpenAI embeddings
"""

import os
from typing import List, Dict, Tuple
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import config
import time


class VectorStore:
    """
    Handles embedding generation and vector storage/retrieval with Pinecone
    """
    
    def __init__(self, api_key: str, index_name: str, openai_api_key: str):
        """
        Initialize the vector store
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            openai_api_key: OpenAI API key
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.index_name = index_name
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        
        # Create or connect to index
        self._initialize_index()
    
    def _initialize_index(self):
        """
        Initialize Pinecone index, create if it doesn't exist
        """
        # Check if index exists
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            time.sleep(1)
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        response = self.openai_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def embed_and_store(self, chunks: List[Tuple[str, dict]]) -> int:
        """
        Generate embeddings for chunks and store in Pinecone
        
        Args:
            chunks: List of tuples containing (chunk_text, metadata)
            
        Returns:
            Number of chunks stored
        """
        vectors_to_upsert = []
        
        for i, (chunk_text, metadata) in enumerate(chunks):
            # Generate embedding
            embedding = self.generate_embedding(chunk_text)
            
            # Create unique ID
            vector_id = f"{metadata['filename']}_{metadata['chunk_index']}_{int(time.time())}"
            
            # Add chunk text to metadata for retrieval
            metadata['text'] = chunk_text
            
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
            
            # Upsert in batches of 100
            if len(vectors_to_upsert) >= 100:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
        
        return len(chunks)
    
    def query_vectors(self, query_text: str, top_k: int = None) -> List[Dict]:
        """
        Query Pinecone for similar vectors with enhanced retrieval
        
        Args:
            query_text: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List of matching results with metadata, sorted by relevance
        """
        if top_k is None:
            top_k = config.TOP_K
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query_text)
        
        # Query Pinecone with higher top_k for better coverage
        results = self.index.query(
            vector=query_embedding,
            top_k=min(top_k * 2, 20),  # Get more results for better filtering
            include_metadata=True
        )
        
        # Extract and return results
        matches = []
        for match in results.matches:
            matches.append({
                'text': match.metadata.get('text', ''),
                'filename': match.metadata.get('filename', ''),
                'score': match.score,
                'chunk_index': match.metadata.get('chunk_index', 0)
            })
        
        # Sort by score (descending) and return top_k
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:top_k]
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index
        
        Returns:
            Dictionary containing index statistics
        """
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension
        }
    
    def delete_all_vectors(self):
        """
        Delete all vectors from the index
        """
        self.index.delete(delete_all=True)

