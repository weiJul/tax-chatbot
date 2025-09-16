"""
LLM Service for RAG Tax System
Handles language model loading, inference, and RAG pipeline integration
Optimized for RTX 2080 Ti
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .config import config_manager
from .embeddings import embedding_service
from .llama_retrieval import hierarchical_retrieval
from .phoenix_tracer import get_phoenix_tracer, initialize_phoenix_tracer
from .query_router import query_router
from .vector_store import vector_store

logger = logging.getLogger(__name__)


class LLMService:
    """
    Language Model Service for RAG system
    Handles model loading, inference, and context management
    """
    
    def __init__(self):
        """Initialize LLM service"""
        self.config = config_manager.llm
        self.rag_config = config_manager.rag
        self.memory_config = config_manager.memory
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = self.config.device if torch.cuda.is_available() else "cpu"
        
        # Query counter for memory management
        self.query_count = 0
        
        # Fallback analytics
        self.fallback_count = 0
        
        # Cache the LangChain wrapper to avoid re-creation
        self._langchain_llm_wrapper = None
        
        # Initialize Phoenix tracing
        self.phoenix_tracer = get_phoenix_tracer()
        if self.phoenix_tracer is None:
            self.phoenix_tracer = initialize_phoenix_tracer(config_manager.config)
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing LLMService with model: {self.config.name}")
    
    def load_model(self) -> None:
        """Load the language model"""
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading LLM: {self.config.name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise
    
    def generate_response(
        self,
        query: str,
        context_documents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response using RAG pipeline
        
        Args:
            query: User query
            context_documents: Retrieved documents for context
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            stream: Whether to stream response
            
        Returns:
            Generated response
        """
        if self.pipeline is None:
            self.load_model()
            
        # Phoenix tracing for LLM generation
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            return self._generate_response_with_tracing(query, context_documents, max_new_tokens, temperature, stream)
        else:
            return self._generate_response_internal(query, context_documents, max_new_tokens, temperature, stream)
    
    def _generate_response_with_tracing(
        self,
        query: str,
        context_documents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """Generate response with Phoenix tracing"""
        with self.phoenix_tracer.tracer.start_as_current_span("llm_generation") as span:
            span.set_attributes(self.phoenix_tracer._create_span_attributes(
                operation_type="llm_generation",
                model_name="GPT-Neo-2.7B",
                query_length=len(query),
                context_length=len(str(context_documents)) if context_documents else 0,
                has_context=bool(context_documents),
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                gpu_optimized="rtx_2080_ti"
            ))
            
            import time
            start_time = time.time()
            
            try:
                response = self._generate_response_internal(query, context_documents, max_new_tokens, temperature, stream)
                
                # Analyze response quality
                response_text = str(response)
                token_count = len(response_text.split())
                
                # Tax-specific quality metrics
                has_citations = any(keyword in response_text.lower() for keyword in 
                                  ['pub29', 'business tax basics', 'irs', 'ftb', 'publication'])
                has_disclaimers = any(keyword in response_text.lower() for keyword in 
                                    ['consult', 'professional', 'advisor', 'may vary'])
                mentions_jurisdiction = any(keyword in response_text.lower() for keyword in 
                                          ['california', 'ca', 'state', 'federal'])
                
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    response_length=len(response_text),
                    token_count=token_count,
                    has_citations=has_citations,
                    has_disclaimers=has_disclaimers,
                    mentions_jurisdiction=mentions_jurisdiction,
                    response_quality_score=self.phoenix_tracer._calculate_response_quality(response_text),
                    generation_success=True
                ))
                
                # Quality alerts
                if token_count < 10:
                    span.set_attribute('quality_alert', 'very_short_response')
                elif token_count > 500:
                    span.set_attribute('quality_alert', 'very_long_response')
                if not has_disclaimers:
                    span.set_attribute('quality_alert', 'missing_tax_disclaimers')
                
                return response
                
            except Exception as e:
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    generation_success=False,
                    error_message=str(e),
                    error_type=type(e).__name__
                ))
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute('duration_ms', duration_ms)
    
    def _generate_response_internal(
        self,
        query: str,
        context_documents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """Internal response generation without tracing"""
        
        # Increment query counter for memory management
        self.query_count += 1
        
        # Build prompt with context
        prompt = self._build_rag_prompt(query, context_documents)
        
        # Generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_full_text": False
        }
        
        try:
            # Generate response
            if stream:
                return self._generate_streaming_response(prompt, generation_kwargs)
            else:
                outputs = self.pipeline(prompt, **generation_kwargs)
                response = outputs[0]["generated_text"].strip()
                
                # Clean up response
                response = self._clean_response(response)
                
                # Memory management
                self._manage_memory()
                
                return response
                
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise
    
    def query_with_rag(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query with retrieval and generation
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            similarity_threshold: Similarity threshold for retrieval
            
        Returns:
            Dictionary with response and metadata
        """
        top_k = top_k or self.rag_config.top_k
        similarity_threshold = similarity_threshold or self.rag_config.similarity_threshold
        
        # Generate query embedding
        query_embedding = embedding_service.embed_query(query)
        
        # Retrieve relevant documents
        documents, metadata, similarities = vector_store.search_with_threshold(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"Retrieved {len(documents)} documents for query")
        
        # Generate response with context
        response = self.generate_response(
            query=query,
            context_documents=documents
        )
        
        return {
            "response": response,
            "query": query,
            "retrieved_documents": documents,
            "document_metadata": metadata,
            "similarities": similarities,
            "num_retrieved": len(documents)
        }
    
    def query_with_hierarchical_rag(self, query: str, jurisdiction: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced RAG query with LlamaIndex hierarchical retrieval
        
        Args:
            query: User query
            jurisdiction: Optional jurisdiction hint for targeted retrieval
            
        Returns:
            Dictionary with response and hierarchical retrieval metadata
        """
        logger.info(f"Processing hierarchical RAG query: '{query[:100]}...'")
        
        try:
            # Ensure hierarchical retrieval is initialized
            if not hierarchical_retrieval.index:
                logger.info("Initializing hierarchical retrieval system...")
                hierarchical_retrieval.initialize()
            
            # Use hierarchical retrieval-only (no generation) with jurisdiction awareness
            if jurisdiction:
                logger.info(f"Using jurisdiction-specific retrieval: {jurisdiction}")
                retrieval_result = hierarchical_retrieval.hierarchical_retrieval_only(query, jurisdiction)
            else:
                # Auto-detect jurisdiction and use retrieval-only
                from ..utils.jurisdiction_detector import jurisdiction_detector
                detected_jurisdiction = jurisdiction_detector.detect_jurisdiction(query)
                logger.info(f"Auto-detected jurisdiction: {detected_jurisdiction}")
                retrieval_result = hierarchical_retrieval.hierarchical_retrieval_only(query, detected_jurisdiction)
            
            # Generate response with our LLM using the retrieved documents
            context_documents = retrieval_result.get("documents", [])
            
            logger.info(f"Generating response with {len(context_documents)} context documents")
            final_response = self.generate_response(
                    query=query,
                    context_documents=context_documents
                )
            
            # Compile comprehensive result
            result = {
                "response": final_response,
                "query": query,
                "retrieval_strategy": retrieval_result.get("retrieval_strategy", "unknown"),
                "jurisdiction_attempted": retrieval_result.get("jurisdiction_attempted"),
                "jurisdiction_success": retrieval_result.get("jurisdiction_success", False),
                "fallback_used": retrieval_result.get("fallback_used", False),
                "num_retrieved": len(retrieval_result.get("documents", [])),
                "retrieved_documents": retrieval_result.get("documents", []),
                "document_metadata": retrieval_result.get("metadata", []),
                "similarities": retrieval_result.get("similarities", []),
                "hierarchical_metadata": retrieval_result
            }
            
            logger.info(f"Hierarchical RAG complete - Strategy: {result['retrieval_strategy']}")
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical RAG failed, falling back to basic RAG: {str(e)}")
            # Fallback to basic RAG
            basic_result = self.query_with_rag(query)
            basic_result["fallback_to_basic"] = True
            basic_result["error"] = str(e)
            return basic_result
    
    def query_with_smart_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Smart query method that automatically uses the best retrieval strategy
        
        Args:
            query: User query
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Try hierarchical retrieval first
            return self.query_with_hierarchical_rag(query)
        except Exception as e:
            logger.warning(f"Hierarchical retrieval failed, using basic RAG: {str(e)}")
            # Fallback to basic RAG
            result = self.query_with_rag(query)
            result["retrieval_method"] = "basic_fallback"
            result["hierarchical_error"] = str(e)
            return result
    
    def _build_rag_prompt(self, query: str, context_documents: Optional[List[str]] = None) -> str:
        """Build RAG prompt with system prompt, context, and query"""
        prompt_parts = []
        
        # Simplified prompt for compatibility with smaller models
        if context_documents:
            # Limit context length to prevent overflow
            context_text = ""
            for i, doc in enumerate(context_documents[:2]):  # Only use top 2 documents
                doc_excerpt = doc[:200] + "..." if len(doc) > 200 else doc  # Limit doc length
                context_text += f"Context {i+1}: {doc_excerpt}\n\n"
            
            prompt_parts.append(f"Context information:\n{context_text}")
        
        # User query
        prompt_parts.append(f"Question: {query}\nAnswer:")
        
        return "".join(prompt_parts)
    
    def _generate_streaming_response(self, prompt: str, generation_kwargs: Dict[str, Any]) -> Generator[str, None, None]:
        """Generate streaming response (placeholder for future implementation)"""
        # For now, return non-streaming response
        # TODO: Implement proper streaming
        outputs = self.pipeline(prompt, **generation_kwargs)
        response = outputs[0]["generated_text"].strip()
        yield self._clean_response(response)
    
    def _clean_response(self, response: str) -> str:
        """Clean generated response"""
        # Remove special tokens and artifacts
        response = response.replace("<|system|>", "")
        response = response.replace("<|context|>", "")
        response = response.replace("<|user|>", "")
        response = response.replace("<|assistant|>", "")
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        return response.strip()
    
    def _manage_memory(self) -> None:
        """Manage GPU memory usage"""
        if self.memory_config.monitor_memory and torch.cuda.is_available():
            if self.query_count % self.memory_config.clear_cache_interval == 0:
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
                self._log_memory_usage()
    
    def _log_memory_usage(self) -> None:
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "model_name": self.config.name,
            "device": self.device,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "status": "loaded",
            "query_count": self.query_count,
            "fallback_count": self.fallback_count
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
            })
        
        return info
    
    def clear_memory(self) -> None:
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
    
    def query_with_router(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Complete query processing with router-based classification
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            similarity_threshold: Similarity threshold for retrieval
            
        Returns:
            Dictionary with response and routing metadata
        """
        # Phoenix tracing for complete RAG pipeline
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            with self.phoenix_tracer.tracer.start_as_current_span("rag_pipeline") as pipeline_span:
                pipeline_span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    operation_type="complete_rag_pipeline",
                    query_text=query[:200],
                    query_length=len(query),
                    interface="llm_service",
                    top_k=top_k or self.rag_config.top_k,
                    similarity_threshold=similarity_threshold or self.rag_config.similarity_threshold
                ))
                
                result = self._execute_router_query(query, top_k, similarity_threshold)
                
                # Add pipeline metrics to span
                pipeline_span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    pipeline_success=True,
                    query_type=result.get('query_type', 'unknown'),
                    context_enhanced=result.get('context_enhanced', False),
                    num_retrieved=result.get('num_retrieved', 0),
                    user_found=bool(result.get('user_context') and result['user_context'].found) if 'user_context' in result else False
                ))
                
                return result
        else:
            return self._execute_router_query(query, top_k, similarity_threshold)
    
    def _execute_router_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        # Initialize router with cached LLM wrapper if not already done
        if query_router.llm is None and self.pipeline is not None:
            if self._langchain_llm_wrapper is None:
                # Create and cache the LLM wrapper once
                self._langchain_llm_wrapper = self._create_langchain_llm()
                logger.info("Created and cached LangChain LLM wrapper")
            query_router.set_llm(self._langchain_llm_wrapper)
        
        # Step 1: Route the query (with Phoenix tracing)
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            with self.phoenix_tracer.tracer.start_as_current_span("query_routing") as routing_span:
                routing_span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    operation_type="query_classification",
                    query_text=query[:200],
                    router_type="langchain_multi_route"
                ))
                routing_result = query_router.route_query(query)
                routing_span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    classification_type=routing_result.get('query_type', 'unknown'),
                    confidence_score=routing_result.get('confidence', 0),
                    classification_success=True
                ))
        else:
            routing_result = query_router.route_query(query)
        
        # Step 2: Process based on new sequential routing decision
        if routing_result['query_type'] == 'general_search_with_context':
            return self._process_context_enhanced_query(query, routing_result, top_k, similarity_threshold)
        else:
            return self._process_general_query(query, routing_result, top_k, similarity_threshold)
    
    def _process_context_enhanced_query(
        self,
        query: str,
        routing_result: Dict[str, Any],
        top_k: Optional[int],
        similarity_threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Process query with enhanced user context from sequential routing"""
        
        # Get user context from successful database lookup
        user_context = routing_result['user_context']
        tax_context = user_context.tax_context
        user_data = user_context.user_data
        
        logger.info(f"Processing context-enhanced query for {user_data.first_name} {user_data.last_name}")
        
        # Enhance query with personal context for retrieval
        enhanced_query = self._enhance_query_with_context(query, tax_context)
        
        # Perform RAG retrieval with enhanced query
        query_embedding = embedding_service.embed_query(enhanced_query)
        documents, metadata, similarities = vector_store.search_with_threshold(
            query_embedding=query_embedding,
            top_k=top_k or self.rag_config.top_k,
            similarity_threshold=similarity_threshold or self.rag_config.similarity_threshold
        )
        
        # Generate personalized response with user context
        response = self._generate_personalized_response(
            query=query,
            context_documents=documents,
            tax_context=tax_context
        )
        
        return {
            "response": response,
            "query": query,
            "query_type": "general_search_with_context",
            "routing_result": routing_result,
            "retrieved_documents": documents,
            "document_metadata": metadata,
            "similarities": similarities,
            "num_retrieved": len(documents),
            "user_context": user_context,
            "context_enhanced": True,
            "user_name": f"{user_data.first_name} {user_data.last_name}"
        }
    
    def _process_general_query(
        self,
        query: str,
        routing_result: Dict[str, Any],
        top_k: Optional[int],
        similarity_threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Process general tax query with hierarchical retrieval"""
        
        # Use hierarchical retrieval with smart auto-detection
        try:
            rag_result = self.query_with_hierarchical_rag(query)
        except Exception as e:
            logger.warning(f"Hierarchical retrieval failed, using basic RAG: {str(e)}")
            # Fallback to standard RAG pipeline
            rag_result = self.query_with_rag(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            rag_result["hierarchical_fallback"] = True
            rag_result["hierarchical_error"] = str(e)
        
        # Add routing information
        rag_result.update({
            "query_type": "general_search",
            "routing_result": routing_result
        })
        
        rag_result["context_enhanced"] = False
        return rag_result
    
    def _enhance_query_with_context(self, query: str, tax_context: Dict[str, Any]) -> str:
        """Enhance query with user's tax context for better retrieval"""
        if not tax_context:
            return query
        
        # Add relevant context terms to improve retrieval
        context_terms = []
        
        if tax_context.get('filing_status'):
            context_terms.append(tax_context['filing_status'])
        
        if tax_context.get('state'):
            context_terms.append(f"{tax_context['state']} state")
        
        if tax_context.get('tax_bracket'):
            context_terms.append(tax_context['tax_bracket'])
        
        if context_terms:
            enhanced = f"{query} {' '.join(context_terms)}"
            logger.info(f"Enhanced query: {enhanced}")
            return enhanced
        
        return query
    
    def _generate_personalized_response(
        self,
        query: str,
        context_documents: List[str],
        tax_context: Dict[str, Any]
    ) -> str:
        """Generate personalized response using user's tax context"""
        
        # Build personalized prompt
        prompt_parts = []
        
        # Add user context
        if tax_context:
            user_info = f"User: {tax_context.get('user_name', 'Unknown')}\n"
            user_info += f"Filing Status: {tax_context.get('filing_status', 'Unknown')}\n"
            user_info += f"Annual Income: ${tax_context.get('annual_income', 0):,.2f}\n"
            user_info += f"State: {tax_context.get('state', 'Unknown')}\n"
            user_info += f"Tax Bracket: {tax_context.get('tax_bracket', 'Unknown')}\n\n"
            
            prompt_parts.append(f"Personal Tax Information:\n{user_info}")
        
        # Add document context
        if context_documents:
            context_text = ""
            for i, doc in enumerate(context_documents[:2]):  # Limit to top 2 docs
                doc_excerpt = doc[:300] + "..." if len(doc) > 300 else doc
                context_text += f"Reference {i+1}: {doc_excerpt}\n\n"
            
            prompt_parts.append(f"Tax Information:\n{context_text}")
        
        # Add personalized instruction
        prompt_parts.append(f"Question: {query}\n")
        prompt_parts.append("Answer this question specifically for the user based on their personal tax information:")
        
        prompt = "".join(prompt_parts)
        
        # Generate response
        try:
            if self.pipeline is None:
                self.load_model()
            
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False
            }
            
            outputs = self.pipeline(prompt, **generation_kwargs)
            response = outputs[0]["generated_text"].strip()
            
            # Clean response
            response = self._clean_response(response)
            
            # Memory management
            self._manage_memory()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate personalized response: {str(e)}")
            return f"I encountered an error generating a personalized response: {str(e)}"
    
    def _create_langchain_llm(self):
        """Create a simple LangChain LLM wrapper for the router"""
        from typing import Any, List, Optional

        from langchain.llms.base import LLM

        # Store pipeline reference for closure
        hf_pipeline = self.pipeline
        
        class HuggingFaceLLMWrapper(LLM):
            """Simple wrapper to make HuggingFace pipeline compatible with LangChain"""
            
            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
            ) -> str:
                """Call the model"""
                try:
                    result = hf_pipeline(
                        prompt,
                        max_new_tokens=100,  # Short responses for classification
                        temperature=0.3,     # Low temperature for consistency
                        do_sample=True,
                        return_full_text=False
                    )
                    return result[0]["generated_text"].strip()
                except Exception as e:
                    logger.error(f"LLM wrapper call failed: {str(e)}")
                    return "general_tax"  # Safe fallback
            
            @property
            def _llm_type(self) -> str:
                return "huggingface_pipeline"
        
        return HuggingFaceLLMWrapper()

    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            self.clear_memory()
            logger.info("Model unloaded")


# Global LLM service instance
llm_service = LLMService()