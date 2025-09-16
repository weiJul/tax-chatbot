"""
Embedding service for RAG Tax System
Handles text embeddings using BGE model optimized for RTX 2080 Ti
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import config_manager
from .phoenix_tracer import get_phoenix_tracer, initialize_phoenix_tracer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using BGE model
    Optimized for RTX 2080 Ti with memory management
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding service"""
        self.config = config_manager.embedding
        self.model_name = model_name or self.config.name
        self.model = None
        self._device = self.config.device if torch.cuda.is_available() else "cpu"
        
        # Initialize Phoenix tracing
        self.phoenix_tracer = get_phoenix_tracer()
        if self.phoenix_tracer is None:
            self.phoenix_tracer = initialize_phoenix_tracer(config_manager)
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
    
    def load_model(self) -> None:
        """Load the embedding model"""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=self.config.cache_dir,
                device=self._device
            )
            
            # Set max sequence length
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.max_seq_length
            
            logger.info(f"Model loaded successfully on device: {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        # Phoenix tracing for embeddings
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            return self._embed_text_with_tracing(text)
        else:
            return self._embed_text_internal(text)
    
    def _embed_text_with_tracing(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings with Phoenix tracing"""
        with self.phoenix_tracer.tracer.start_as_current_span("embeddings") as span:
            # Extract input information
            if isinstance(text, str):
                batch_size = 1
                input_length = len(text)
                texts = [text]
            else:
                batch_size = len(text)
                input_length = sum(len(t) for t in text)
                texts = text
            
            span.set_attributes(self.phoenix_tracer._create_span_attributes(
                operation_type='embeddings',
                embedding_model='BGE-base-en-v1.5',
                batch_size=batch_size,
                input_length=input_length,
                gpu_optimized=True,
                device=self._device
            ))
            
            import time
            start_time = time.time()
            
            try:
                result = self._embed_text_internal(text)
                
                # Extract result metrics
                if hasattr(result, 'shape'):
                    embedding_dim = result.shape[-1] if len(result.shape) > 1 else result.shape[0]
                    span.set_attributes(self.phoenix_tracer._create_span_attributes(
                        embedding_dimension=embedding_dim,
                        result_shape=str(result.shape),
                        embedding_success=True
                    ))
                
                return result
                
            except Exception as e:
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    embedding_success=False,
                    error_message=str(e),
                    error_type=type(e).__name__
                ))
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute('duration_ms', duration_ms)
                
                # Performance alerts for RTX 2080 Ti
                if duration_ms > 1000:  # Alert if embedding takes >1s
                    span.set_attribute('performance_alert', 'slow_embedding')
    
    def _embed_text_internal(self, text: Union[str, List[str]]) -> np.ndarray:
        """Internal embedding generation without tracing"""
        # Ensure input is a list
        if isinstance(text, str):
            texts = [text]
            single_text = True
        else:
            texts = text
            single_text = False
        
        try:
            # Generate embeddings in batches for memory efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Return single embedding if single text was provided
            if single_text:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of document texts
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")
        
        embeddings = self.embed_text(documents)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query
        
        Args:
            query: Query text
            
        Returns:
            numpy array embedding
        """
        return self.embed_text(query)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return np.dot(embedding1, embedding2)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self._device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "status": "loaded"
        }
    
    def clear_memory(self) -> None:
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.clear_memory()
            logger.info("Model unloaded")


# Global embedding service instance
embedding_service = EmbeddingService()