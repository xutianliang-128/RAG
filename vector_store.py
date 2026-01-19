"""
Vector database creation and retrieval module
Uses ChromaDB as vector storage
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os


class VectorStore:
    """Vector storage class"""
    
    def __init__(self, collection_name: str = "meeting_documents", persist_directory: str = "./chroma_db"):
        """
        Initialize vector store
        
        Args:
            collection_name: Collection name
            persist_directory: Persistence directory
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model (using lightweight sentence-transformers)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 5000):
        """
        Add documents to vector database
        
        Args:
            documents: List of documents, each containing 'text' and 'metadata'
            batch_size: Maximum number of documents to add in each batch (ChromaDB limit)
        """
        total_docs = len(documents)
        print(f"Adding {total_docs} documents to vector database (in batches of {batch_size})...")
        
        total_added = 0
        
        # Process in batches to avoid ChromaDB batch size limit
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs - 1) // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            
            # Generate embedding vectors
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            total_added += len(batch)
            print(f"  Added {len(batch)} documents (total: {total_added}/{total_docs})")
        
        print(f"Successfully added {total_added} documents")
    
    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            n_results: Number of results to return
            filter_dict: Optional filter conditions (e.g., {'is_public_comment': 1})
            
        Returns:
            List of similar documents containing text, metadata, and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build query
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        
        # Add filter conditions
        if filter_dict:
            # ChromaDB uses where clause for filtering
            where_clause = {}
            for key, value in filter_dict.items():
                where_clause[key] = value
            query_kwargs["where"] = where_clause
        
        # Execute query
        results = self.collection.query(**query_kwargs)
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_collection_size(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()
