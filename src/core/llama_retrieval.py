"""
LlamaIndex Hierarchical Retrieval Service
Wraps existing ChromaDB + BGE embeddings with LlamaIndex for advanced retrieval
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from ..utils.jurisdiction_detector import jurisdiction_detector
from .config import config_manager
from .phoenix_tracer import get_phoenix_tracer, initialize_phoenix_tracer

logger = logging.getLogger(__name__)


class HierarchicalRetrieval:
    """
    LlamaIndex-based hierarchical retrieval system
    Wraps existing ChromaDB with jurisdiction-aware smart retrieval
    """
    
    def __init__(self):
        """Initialize hierarchical retrieval system"""
        self.config = config_manager.vector_store
        self.rag_config = config_manager.rag
        
        # Initialize components
        self.chroma_client = None
        self.chroma_collection = None
        self.vector_store = None
        self.index = None
        self.embedding_model = None
        
        # Query engines for different strategies
        self.general_query_engine = None
        self.jurisdiction_query_engines = {}
        
        # Initialize Phoenix tracing
        self.phoenix_tracer = get_phoenix_tracer()
        if self.phoenix_tracer is None:
            self.phoenix_tracer = initialize_phoenix_tracer(config_manager.config)
        
        logger.info("Initializing LlamaIndex hierarchical retrieval system")
    
    def initialize(self) -> None:
        """Initialize all components"""
        try:
            # Initialize embedding model (BGE)
            self._initialize_embeddings()
            
            # Initialize LLM for LlamaIndex
            self._initialize_llm()
            
            # Initialize ChromaDB connection
            self._initialize_chroma()
            
            # Initialize LlamaIndex components
            self._initialize_llamaindex()
            
            # Initialize query engines
            self._initialize_query_engines()
            
            logger.info("Hierarchical retrieval system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hierarchical retrieval: {str(e)}")
            raise
    
    def _initialize_embeddings(self) -> None:
        """Initialize BGE embedding model through LlamaIndex"""
        try:
            embedding_config = config_manager.embedding
            
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embedding_config.name,
                cache_folder=embedding_config.cache_dir,
                max_length=embedding_config.max_seq_length,
                device=embedding_config.device
            )
            
            # Set as global embedding model for LlamaIndex
            Settings.embed_model = self.embedding_model
            
            logger.info(f"Initialized BGE embedding model: {embedding_config.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize LLM for LlamaIndex to avoid OpenAI dependency"""
        try:
            # For now, disable LLM completely to focus on retrieval
            # The system will use the existing LLM service for generation
            Settings.llm = None
            
            logger.info("LlamaIndex LLM disabled - using retrieval-only mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB connection"""
        if chromadb is None:
            raise ImportError("chromadb is required for vector storage")
        
        try:
            # Connect to existing ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "description": "Tax document embeddings with jurisdiction metadata",
                    "distance_metric": self.config.distance_metric
                }
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _initialize_llamaindex(self) -> None:
        """Initialize LlamaIndex components"""
        try:
            # Create ChromaVectorStore wrapper
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            
            # Create VectorStoreIndex
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_model
            )
            
            logger.info("LlamaIndex components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex: {str(e)}")
            raise
    
    def _initialize_query_engines(self) -> None:
        """Initialize query engines for different retrieval strategies"""
        try:
            similarity_threshold = self.rag_config.similarity_threshold
            top_k = self.rag_config.top_k
            
            # General query engine (no filters)
            self.general_query_engine = self._create_query_engine(
                filters=None,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # California-specific query engine
            self.jurisdiction_query_engines["california"] = self._create_query_engine(
                filters={"jurisdiction": "california"},
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            logger.info("Query engines initialized for general and California jurisdiction")
            
        except Exception as e:
            logger.error(f"Failed to initialize query engines: {str(e)}")
            raise
    
    def _create_query_engine(
        self, 
        filters: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.4
    ) -> BaseQueryEngine:
        """
        Create a query engine with optional metadata filters
        
        Args:
            filters: Metadata filters to apply
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Configured query engine
        """
        try:
            # Create retriever with filters
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
                filters=self._convert_filters_to_llamaindex(filters) if filters else None
            )
            
            # Add similarity postprocessor
            postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=similarity_threshold)
            ]
            
            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=postprocessors
            )
            
            return query_engine
            
        except Exception as e:
            logger.error(f"Failed to create query engine: {str(e)}")
            raise
    
    def _convert_filters_to_llamaindex(self, filters: Dict[str, str]) -> Optional[Any]:
        """Convert simple filters to LlamaIndex filter format"""
        if not filters:
            return None
        
        try:
            from llama_index.core.vector_stores.types import (ExactMatchFilter,
                                                              MetadataFilters)
            
            filter_list = []
            for key, value in filters.items():
                filter_list.append(ExactMatchFilter(key=key, value=value))
            
            return MetadataFilters(filters=filter_list)
            
        except ImportError:
            # Fallback for older LlamaIndex versions
            logger.warning("Using simple dict filters - upgrade LlamaIndex for advanced filtering")
            return filters
        except Exception as e:
            logger.error(f"Failed to convert filters: {str(e)}")
            return None
    
    def hierarchical_query(self, user_query: str, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """
        Core hierarchical query function with smart retrieval
        
        Args:
            user_query: User's query text
            jurisdiction: Optional jurisdiction hint (e.g., "california")
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.index:
            raise RuntimeError("Hierarchical retrieval not initialized. Call initialize() first.")
        
        # Phoenix tracing for hierarchical retrieval
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            return self._hierarchical_query_with_tracing(user_query, jurisdiction)
        else:
            return self._hierarchical_query_internal(user_query, jurisdiction)
    
    def _hierarchical_query_with_tracing(self, user_query: str, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """Hierarchical query with Phoenix tracing"""
        with self.phoenix_tracer.tracer.start_as_current_span("hierarchical_retrieval") as span:
            span.set_attributes(self.phoenix_tracer._create_span_attributes(
                operation_type="hierarchical_retrieval",
                query_text=user_query[:200],
                query_length=len(user_query),
                jurisdiction=jurisdiction or "auto-detect",
                retrieval_engine="llamaindex_wrapper",
                vector_store="chromadb"
            ))
            
            import time
            start_time = time.time()
            
            try:
                result = self._hierarchical_query_internal(user_query, jurisdiction)
                
                # Extract hierarchical retrieval metrics
                num_documents = len(result.get('source_nodes', []))
                
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    num_documents_retrieved=num_documents,
                    jurisdiction_attempted=result.get('jurisdiction_attempted'),
                    jurisdiction_success=result.get('jurisdiction_success', False),
                    fallback_used=result.get('fallback_used', False),
                    retrieval_strategy=result.get('retrieval_strategy', 'unknown'),
                    hierarchical_success=True
                ))
                
                # Quality alerts
                if num_documents == 0:
                    span.set_attribute('quality_alert', 'no_documents_retrieved')
                elif not result.get('jurisdiction_success', False) and jurisdiction:
                    span.set_attribute('quality_alert', 'jurisdiction_specific_failed')
                
                return result
                
            except Exception as e:
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    hierarchical_success=False,
                    error_message=str(e),
                    error_type=type(e).__name__
                ))
                raise
                
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute('duration_ms', duration_ms)
    
    def _hierarchical_query_internal(self, user_query: str, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """Internal hierarchical query implementation without tracing"""
        logger.info(f"Processing hierarchical query: '{user_query[:100]}...' (jurisdiction: {jurisdiction})")
        
        result = {
            "user_query": user_query,
            "jurisdiction_attempted": jurisdiction,
            "jurisdiction_success": False,
            "fallback_used": False,
            "response": None,
            "source_nodes": [],
            "retrieval_strategy": "unknown"
        }
        
        try:
            # Step 1: Try jurisdiction-specific retrieval if jurisdiction provided
            if jurisdiction and jurisdiction in self.jurisdiction_query_engines:
                logger.debug(f"Attempting {jurisdiction}-specific retrieval")
                
                jurisdiction_engine = self.jurisdiction_query_engines[jurisdiction]
                jurisdiction_response = jurisdiction_engine.query(user_query)
                
                # Check if we got meaningful results
                if jurisdiction_response.response and jurisdiction_response.response.strip():
                    result["jurisdiction_success"] = True
                    result["response"] = jurisdiction_response.response
                    result["source_nodes"] = jurisdiction_response.source_nodes
                    result["retrieval_strategy"] = f"{jurisdiction}_specific"
                    
                    logger.info(f"Successfully retrieved from {jurisdiction} jurisdiction")
                    return result
            
            # Step 2: Fallback to general retrieval
            logger.debug("Using general retrieval (no jurisdiction filter)")
            
            general_response = self.general_query_engine.query(user_query)
            
            result["fallback_used"] = True
            result["response"] = general_response.response
            result["source_nodes"] = general_response.source_nodes
            result["retrieval_strategy"] = "general_fallback"
            
            logger.info("Retrieved using general fallback")
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical query failed: {str(e)}")
            result["error"] = str(e)
            return result
    
    def hierarchical_retrieval_only(self, user_query: str, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """
        Hierarchical retrieval without text generation - returns only retrieved documents
        
        Args:
            user_query: User's query text
            jurisdiction: Optional jurisdiction hint (e.g., "california")
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        if not self.index:
            raise RuntimeError("Hierarchical retrieval not initialized. Call initialize() first.")
        
        logger.info(f"Processing hierarchical retrieval: '{user_query[:100]}...' (jurisdiction: {jurisdiction})")
        
        result = {
            "user_query": user_query,
            "jurisdiction_attempted": jurisdiction,
            "jurisdiction_success": False,
            "fallback_used": False,
            "documents": [],
            "metadata": [],
            "similarities": [],
            "source_nodes": [],
            "retrieval_strategy": "unknown"
        }
        
        try:
            # Step 1: Try jurisdiction-specific retrieval if jurisdiction provided
            if jurisdiction and jurisdiction in self.jurisdiction_query_engines:
                logger.debug(f"Attempting {jurisdiction}-specific retrieval")
                
                retriever = self.jurisdiction_query_engines[jurisdiction].retriever
                retrieved_nodes = retriever.retrieve(user_query)
                
                # Check if we got meaningful results
                if retrieved_nodes and len(retrieved_nodes) > 0:
                    result["jurisdiction_success"] = True
                    result["source_nodes"] = retrieved_nodes
                    result["retrieval_strategy"] = f"{jurisdiction}_specific"
                    
                    # Extract documents and metadata
                    result["documents"] = [node.text for node in retrieved_nodes]
                    result["metadata"] = [node.metadata for node in retrieved_nodes]
                    result["similarities"] = [node.score if hasattr(node, 'score') else 0.0 for node in retrieved_nodes]
                    
                    logger.info(f"Successfully retrieved {len(retrieved_nodes)} documents from {jurisdiction} jurisdiction")
                    return result
            
            # Step 2: Fallback to general retrieval
            logger.debug("Using general retrieval (no jurisdiction filter)")
            
            retriever = self.general_query_engine.retriever
            retrieved_nodes = retriever.retrieve(user_query)
            
            result["fallback_used"] = True
            result["source_nodes"] = retrieved_nodes
            result["retrieval_strategy"] = "general_fallback"
            
            # Extract documents and metadata
            result["documents"] = [node.text for node in retrieved_nodes]
            result["metadata"] = [node.metadata for node in retrieved_nodes]
            result["similarities"] = [node.score if hasattr(node, 'score') else 0.0 for node in retrieved_nodes]
            
            logger.info(f"Retrieved {len(retrieved_nodes)} documents using general fallback")
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {str(e)}")
            result["error"] = str(e)
            return result
    
    def smart_query(self, user_query: str) -> Dict[str, Any]:
        """
        Smart query with automatic jurisdiction detection
        
        Args:
            user_query: User's query text
            
        Returns:
            Dictionary with response and metadata
        """
        # Auto-detect jurisdiction
        jurisdiction = jurisdiction_detector.detect_jurisdiction(user_query)
        
        # Get confidence scores for logging
        confidence_scores = jurisdiction_detector.get_jurisdiction_confidence(user_query)
        
        logger.info(f"Auto-detected jurisdiction: {jurisdiction} (confidence: {confidence_scores})")
        
        # Perform hierarchical query
        result = self.hierarchical_query(user_query, jurisdiction)
        result["auto_detected_jurisdiction"] = jurisdiction
        result["jurisdiction_confidence"] = confidence_scores
        
        return result
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        try:
            stats = {
                "collection_name": self.config.collection_name,
                "total_documents": 0,
                "index_initialized": self.index is not None,
                "query_engines_available": list(self.jurisdiction_query_engines.keys()) + ["general"],
                "embedding_model": config_manager.embedding.name,
            }
            
            if self.chroma_collection:
                stats["total_documents"] = self.chroma_collection.count()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {str(e)}")
            return {"error": str(e)}
    
    def check_jurisdiction_data(self, jurisdiction: str) -> Dict[str, Any]:
        """
        Check if jurisdiction-specific data exists
        
        Args:
            jurisdiction: Jurisdiction to check (e.g., "california")
            
        Returns:
            Dictionary with jurisdiction data statistics
        """
        if not self.chroma_collection:
            return {"error": "ChromaDB not initialized"}
        
        try:
            # Query with jurisdiction filter to count documents
            results = self.chroma_collection.query(
                query_texts=["tax"],  # Generic query
                n_results=1,
                where={"jurisdiction": jurisdiction}
            )
            
            jurisdiction_count = len(results["documents"][0]) if results["documents"] else 0
            
            return {
                "jurisdiction": jurisdiction,
                "document_count": jurisdiction_count,
                "has_data": jurisdiction_count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to check jurisdiction data: {str(e)}")
            return {"jurisdiction": jurisdiction, "error": str(e)}


# Global hierarchical retrieval instance
hierarchical_retrieval = HierarchicalRetrieval()