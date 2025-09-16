"""
Vector store implementation using ChromaDB for RAG Tax System
"""
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings

from .config import config_manager
from .phoenix_tracer import get_phoenix_tracer, initialize_phoenix_tracer

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store implementation using ChromaDB
    Handles document storage, retrieval, and similarity search
    """
    
    def __init__(self):
        """Initialize vector store"""
        self.config = config_manager.vector_store
        self.client = None
        self.collection = None
        
        # Initialize Phoenix tracing
        self.phoenix_tracer = get_phoenix_tracer()
        if self.phoenix_tracer is None:
            self.phoenix_tracer = initialize_phoenix_tracer(config_manager.config)
        
        # Create persist directory
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing VectorStore with collection: {self.config.collection_name}")
    
    def connect(self) -> None:
        """Connect to ChromaDB"""
        if self.client is not None:
            logger.info("Already connected to ChromaDB")
            return
        
        try:
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "description": "Tax document embeddings",
                    "distance_metric": self.config.distance_metric
                }
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            embeddings: Document embeddings array
            metadata: Optional metadata for each document
            ids: Optional document IDs (will generate if not provided)
            
        Returns:
            List of document IDs
        """
        if self.collection is None:
            self.connect()
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare metadata (filter out complex types that ChromaDB doesn't support)
        if metadata is None:
            metadata = [{} for _ in documents]
        else:
            metadata = [self._filter_metadata_for_chromadb(meta) for meta in metadata]
        
        # Ensure embeddings are in the right format
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise
    
    def _filter_metadata_for_chromadb(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to only include types supported by ChromaDB
        ChromaDB supports: str, int, float, bool, None
        """
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, list):
                # Skip lists/complex structures - log for debugging
                logger.debug(f"Skipping complex metadata field '{key}' of type {type(value)}")
            else:
                # Try to convert to string as fallback
                try:
                    filtered[key] = str(value)
                except:
                    logger.debug(f"Could not convert metadata field '{key}' to supported type")
        return filtered
    
    def add_single_document(
        self,
        document: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a single document to the vector store
        
        Args:
            document: Document text
            embedding: Document embedding
            metadata: Optional metadata
            doc_id: Optional document ID
            
        Returns:
            Document ID
        """
        return self.add_documents(
            documents=[document],
            embeddings=embedding.reshape(1, -1),
            metadata=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None
        )[0]
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Tuple of (documents, metadata, distances)
        """
        if self.collection is None:
            self.connect()
        
        # Phoenix tracing for vector search
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            return self._search_with_tracing(query_embedding, top_k, where_filter)
        else:
            return self._search_internal(query_embedding, top_k, where_filter)
    
    def _search_with_tracing(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Search with Phoenix tracing"""
        with self.phoenix_tracer.tracer.start_as_current_span("vector_search") as span:
            # Extract search parameters
            jurisdiction_filter = where_filter.get('jurisdiction') if where_filter else None
            
            span.set_attributes(self.phoenix_tracer._create_span_attributes(
                operation_type="vector_search",
                vector_store="chromadb",
                top_k=top_k,
                has_filter=bool(where_filter),
                jurisdiction_filter=jurisdiction_filter,
                embedding_dimension=len(query_embedding) if hasattr(query_embedding, '__len__') else 'unknown'
            ))
            
            import time
            start_time = time.time()
            
            try:
                documents, metadata, similarities = self._search_internal(query_embedding, top_k, where_filter)
                
                # Extract search quality metrics
                jurisdictions_found = []
                document_sources = []
                
                for meta in metadata:
                    if isinstance(meta, dict):
                        jurisdictions_found.append(meta.get('jurisdiction', 'unknown'))
                        document_sources.append(meta.get('source', 'unknown'))
                
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    num_documents_found=len(documents),
                    jurisdictions_found=list(set(jurisdictions_found)),
                    document_sources=list(set(document_sources)),
                    max_similarity=max(similarities) if similarities else 0,
                    min_similarity=min(similarities) if similarities else 0,
                    avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                    search_success=True
                ))
                
                # Quality alerts
                if len(documents) == 0:
                    span.set_attribute('quality_alert', 'no_documents_found')
                elif similarities and max(similarities) < 0.7:
                    span.set_attribute('quality_alert', 'low_similarity_scores')
                
                return documents, metadata, similarities
                
            except Exception as e:
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    search_success=False,
                    error_message=str(e),
                    error_type=type(e).__name__
                ))
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute('duration_ms', duration_ms)
    
    def _search_internal(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Internal search implementation without tracing"""
        # Ensure embedding is in the right format
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract results
            documents = results["documents"][0] if results["documents"] else []
            metadata = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            # Convert distances to similarities (ChromaDB uses distance, we want similarity)
            similarities = [1 - dist for dist in distances]
            
            logger.info(f"Found {len(documents)} similar documents")
            return documents, metadata, similarities
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {str(e)}")
            raise
    
    def search_with_threshold(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for similar documents with similarity threshold
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            where_filter: Optional metadata filter
            
        Returns:
            Tuple of (documents, metadata, similarities) above threshold
        """
        documents, metadata, similarities = self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where_filter=where_filter
        )
        
        # Filter by threshold
        filtered_results = [
            (doc, meta, sim) for doc, meta, sim in zip(documents, metadata, similarities)
            if sim >= similarity_threshold
        ]
        
        if filtered_results:
            filtered_documents, filtered_metadata, filtered_similarities = zip(*filtered_results)
            return list(filtered_documents), list(filtered_metadata), list(filtered_similarities)
        else:
            return [], [], []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if self.collection is None:
            self.connect()
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.config.collection_name,
                "document_count": count,
                "persist_directory": self.config.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        if self.client is None:
            self.connect()
        
        try:
            self.client.delete_collection(self.config.collection_name)
            self.collection = None
            logger.info(f"Deleted collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all documents)"""
        if self.collection is None:
            self.connect()
        
        try:
            # Delete all documents in the collection
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            
            logger.info(f"Reset collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise
    
    def update_document(
        self,
        doc_id: str,
        document: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update an existing document"""
        if self.collection is None:
            self.connect()
        
        update_data = {"ids": [doc_id]}
        
        if document is not None:
            update_data["documents"] = [document]
        
        if embedding is not None:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            update_data["embeddings"] = [embedding]
        
        if metadata is not None:
            update_data["metadatas"] = [metadata]
        
        try:
            self.collection.update(**update_data)
            logger.info(f"Updated document: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {str(e)}")
            raise
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document by ID"""
        if self.collection is None:
            self.connect()
        
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            raise


# Global vector store instance
vector_store = VectorStore()