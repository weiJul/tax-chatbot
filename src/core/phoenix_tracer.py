"""
Phoenix-Arize tracing instrumentation for Tax RAG system
Provides comprehensive observability for LlamaIndex hierarchical retrieval,
jurisdiction-aware routing, and tax-specific evaluation metrics.
"""
import functools
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    import phoenix as px
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import \
        OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    PHOENIX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Phoenix not available: {e}. Tracing will be disabled.")
    PHOENIX_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaxRagTracer:
    """
    Custom Phoenix tracer for tax RAG system with specialized tax domain monitoring.
    
    Features:
    - Hierarchical retrieval tracing (jurisdiction-aware)
    - Tax accuracy metrics and evaluation
    - Performance monitoring for RTX 2080 Ti constraints
    - User context enhancement tracking
    - Query classification monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.phoenix_session = None
        self.tracer = None
        
        # Handle both dict config and ConfigManager object
        if hasattr(config, 'phoenix'):
            phoenix_config = config.phoenix
            if hasattr(phoenix_config, 'tracing'):
                tracing_config = phoenix_config.tracing
                self.enabled = PHOENIX_AVAILABLE and getattr(tracing_config, 'enabled', True)
            else:
                self.enabled = PHOENIX_AVAILABLE
        else:
            self.enabled = PHOENIX_AVAILABLE and self.config.get('phoenix', {}).get('tracing', {}).get('enabled', True)
        
        if self.enabled:
            self.initialize()
        else:
            logger.info("Phoenix tracing is disabled")
    
    def initialize(self):
        """Initialize Phoenix tracing with tax RAG specific configuration"""
        if not PHOENIX_AVAILABLE:
            logger.warning("Phoenix not available - tracing disabled")
            return
        
        try:
            # Handle both dict config and ConfigManager object
            if hasattr(self.config, 'phoenix'):
                phoenix_config = self.config.phoenix
                server_config = phoenix_config.server if hasattr(phoenix_config, 'server') else {}
                tracing_config = phoenix_config.tracing if hasattr(phoenix_config, 'tracing') else {}
                
                # Launch Phoenix server
                host = getattr(server_config, 'host', 'localhost')
                port = getattr(server_config, 'port', 6006)
                grpc_port = getattr(server_config, 'grpc_port', 6007)
                
                # Set up OpenTelemetry tracer
                project_name = getattr(tracing_config, 'project_name', 'tax-rag-system')
            else:
                # Fallback for dict config
                phoenix_config = self.config.get('phoenix', {})
                server_config = phoenix_config.get('server', {})
                tracing_config = phoenix_config.get('tracing', {})
                
                host = server_config.get('host', 'localhost')
                port = server_config.get('port', 6006)
                grpc_port = server_config.get('grpc_port', 6007)
                project_name = tracing_config.get('project_name', 'tax-rag-system')
            
            # Set environment variables for Phoenix (recommended approach)
            import os
            os.environ['PHOENIX_HOST'] = host
            os.environ['PHOENIX_PORT'] = str(port)
            os.environ['PHOENIX_GRPC_PORT'] = str(grpc_port)
            
            # Check if Phoenix is already running on this port
            import socket
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex((host, port))
                    if result == 0:
                        logger.info(f"Phoenix already running on {host}:{port}, reusing existing session")
                        self.phoenix_session = None  # Don't create new session
                    else:
                        logger.info(f"Starting Phoenix server on {host}:{port}")
                        self.phoenix_session = px.launch_app()
            except Exception as e:
                logger.warning(f"Port check failed, attempting to start Phoenix anyway: {e}")
                self.phoenix_session = px.launch_app()
            
            self.tracer = otel_trace.get_tracer(project_name)
            
            # Instrument requests for HTTP tracing
            RequestsInstrumentor().instrument()
            
            logger.info(f"Phoenix tracing initialized successfully - Dashboard: http://{host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix tracing: {e}")
            self.enabled = False
    
    def _create_span_attributes(self, **kwargs) -> Dict[str, Any]:
        """Create standardized span attributes for tax RAG operations"""
        base_attributes = {
            'service.name': 'tax-rag-system',
            'service.version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add custom attributes, converting complex types to strings
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                base_attributes[key] = value
            else:
                base_attributes[key] = str(value)
                
        return base_attributes
    
    def trace_embeddings(self, func):
        """Decorator for BGE embedding operations with performance monitoring"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("embeddings") as span:
                start_time = time.time()
                
                # Extract input information
                input_text = args[1] if len(args) > 1 else kwargs.get('text', '')
                batch_size = len(input_text) if isinstance(input_text, list) else 1
                
                span.set_attributes(self._create_span_attributes(
                    operation_type='embeddings',
                    embedding_model='BGE-base-en-v1.5',
                    batch_size=batch_size,
                    input_length=len(str(input_text)),
                    gpu_optimized=True
                ))
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Extract result metrics
                    if hasattr(result, 'shape'):
                        embedding_dim = result.shape[-1] if len(result.shape) > 1 else result.shape[0]
                        span.set_attributes(self._create_span_attributes(
                            embedding_dimension=embedding_dim,
                            result_shape=str(result.shape),
                            success=True
                        ))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attributes(self._create_span_attributes(
                        success=False,
                        error_message=str(e),
                        error_type=type(e).__name__
                    ))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute('duration_ms', duration_ms)
                    
                    # Performance alerts for RTX 2080 Ti
                    if duration_ms > 1000:  # Alert if embedding takes >1s
                        span.set_attribute('performance_alert', 'slow_embedding')
        
        return wrapper
    
    def trace_retrieval(self, func):
        """Decorator for hierarchical retrieval operations with jurisdiction tracking"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("retrieval") as span:
                start_time = time.time()
                
                # Extract query information
                query = args[1] if len(args) > 1 else kwargs.get('query', kwargs.get('user_query', ''))
                jurisdiction = kwargs.get('jurisdiction', 'auto-detect')
                
                span.set_attributes(self._create_span_attributes(
                    operation_type='hierarchical_retrieval',
                    query_text=str(query)[:200],  # Truncate long queries
                    query_length=len(str(query)),
                    jurisdiction=jurisdiction,
                    retrieval_engine='llamaindex_wrapper',
                    vector_store='chromadb'
                ))
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Extract retrieval metrics from LlamaIndex response
                    if hasattr(result, 'source_nodes'):
                        num_documents = len(result.source_nodes)
                        
                        # Extract similarity scores and metadata
                        similarities = []
                        jurisdictions_found = []
                        document_sources = []
                        
                        for node in result.source_nodes:
                            if hasattr(node, 'score'):
                                similarities.append(node.score)
                            if hasattr(node, 'metadata'):
                                metadata = node.metadata or {}
                                jurisdictions_found.append(metadata.get('jurisdiction', 'unknown'))
                                document_sources.append(metadata.get('source', 'unknown'))
                        
                        span.set_attributes(self._create_span_attributes(
                            num_documents_retrieved=num_documents,
                            jurisdictions_found=list(set(jurisdictions_found)),
                            document_sources=list(set(document_sources)),
                            max_similarity=max(similarities) if similarities else 0,
                            min_similarity=min(similarities) if similarities else 0,
                            avg_similarity=sum(similarities) / len(similarities) if similarities else 0,
                            retrieval_strategy='hierarchical_fallback'
                        ))
                        
                        # Quality alerts
                        if num_documents == 0:
                            span.set_attribute('quality_alert', 'no_documents_found')
                        elif similarities and max(similarities) < 0.7:
                            span.set_attribute('quality_alert', 'low_similarity_scores')
                    
                    # Handle dict-style results (from legacy retrieval)
                    elif isinstance(result, dict):
                        span.set_attributes(self._create_span_attributes(
                            num_documents_retrieved=len(result.get('documents', [])),
                            jurisdiction=result.get('jurisdiction', 'unknown'),
                            strategy=result.get('strategy', 'unknown')
                        ))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attributes(self._create_span_attributes(
                        success=False,
                        error_message=str(e),
                        error_type=type(e).__name__
                    ))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute('duration_ms', duration_ms)
        
        return wrapper
    
    def trace_llm_generation(self, func):
        """Decorator for LLM generation with context enhancement tracking"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("llm_generation") as span:
                start_time = time.time()
                
                # Extract generation parameters
                query = kwargs.get('query', args[1] if len(args) > 1 else '')
                context = kwargs.get('context', '')
                user_context = kwargs.get('user_context', None)
                
                span.set_attributes(self._create_span_attributes(
                    operation_type='llm_generation',
                    model_name='GPT-Neo-2.7B',
                    query_length=len(str(query)),
                    context_length=len(str(context)),
                    has_user_context=user_context is not None and getattr(user_context, 'found', False),
                    gpu_optimized='rtx_2080_ti'
                ))
                
                try:
                    response = func(*args, **kwargs)
                    
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
                    
                    span.set_attributes(self._create_span_attributes(
                        response_length=len(response_text),
                        token_count=token_count,
                        has_citations=has_citations,
                        has_disclaimers=has_disclaimers,
                        mentions_jurisdiction=mentions_jurisdiction,
                        response_quality_score=self._calculate_response_quality(response_text),
                        success=True
                    ))
                    
                    # Quality alerts
                    if token_count < 10:
                        span.set_attribute('quality_alert', 'very_short_response')
                    elif token_count > 500:
                        span.set_attribute('quality_alert', 'very_long_response')
                    if not has_disclaimers:
                        span.set_attribute('quality_alert', 'missing_tax_disclaimers')
                    
                    span.set_status(Status(StatusCode.OK))
                    return response
                    
                except Exception as e:
                    span.set_attributes(self._create_span_attributes(
                        success=False,
                        error_message=str(e),
                        error_type=type(e).__name__
                    ))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute('duration_ms', duration_ms)
        
        return wrapper
    
    def trace_query_routing(self, func):
        """Decorator for query classification and routing operations"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("query_routing") as span:
                start_time = time.time()
                
                query = args[1] if len(args) > 1 else kwargs.get('query', '')
                
                span.set_attributes(self._create_span_attributes(
                    operation_type='query_classification',
                    query_text=str(query)[:200],
                    router_type='langchain_multi_route'
                ))
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Extract classification results
                    if hasattr(result, 'query_type'):
                        query_type = result.query_type
                        confidence = getattr(result, 'confidence', 0)
                    elif isinstance(result, dict):
                        query_type = result.get('query_type', 'unknown')
                        confidence = result.get('confidence', 0)
                    else:
                        query_type = 'unknown'
                        confidence = 0
                    
                    span.set_attributes(self._create_span_attributes(
                        classification_type=query_type,
                        confidence_score=confidence,
                        is_personal_query=query_type == 'user_data',
                        classification_accuracy=confidence,
                        success=True
                    ))
                    
                    # Quality alerts
                    if confidence < 0.7:
                        span.set_attribute('quality_alert', 'low_classification_confidence')
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attributes(self._create_span_attributes(
                        success=False,
                        error_message=str(e),
                        error_type=type(e).__name__
                    ))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute('duration_ms', duration_ms)
        
        return wrapper
    
    def trace_user_lookup(self, func):
        """Decorator for MCP database user lookup operations"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("user_lookup") as span:
                start_time = time.time()
                
                # Extract lookup parameters
                lookup_method = func.__name__
                lookup_params = {}
                if 'first_name' in kwargs:
                    lookup_params['lookup_type'] = 'name'
                    lookup_params['has_first_name'] = bool(kwargs.get('first_name'))
                    lookup_params['has_last_name'] = bool(kwargs.get('last_name'))
                elif 'email' in kwargs:
                    lookup_params['lookup_type'] = 'email'
                elif 'tax_id' in kwargs:
                    lookup_params['lookup_type'] = 'tax_id'
                
                span.set_attributes(self._create_span_attributes(
                    operation_type='user_lookup',
                    lookup_method=lookup_method,
                    mcp_server='tax_database',
                    **lookup_params
                ))
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Analyze lookup result
                    user_found = result is not None
                    if user_found and hasattr(result, 'first_name'):
                        span.set_attributes(self._create_span_attributes(
                            user_found=True,
                            has_tax_data=hasattr(result, 'annual_income'),
                            filing_status=getattr(result, 'filing_status', 'unknown'),
                            state=getattr(result, 'state', 'unknown')
                        ))
                    else:
                        span.set_attribute('user_found', False)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_attributes(self._create_span_attributes(
                        success=False,
                        error_message=str(e),
                        error_type=type(e).__name__
                    ))
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                    
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute('duration_ms', duration_ms)
        
        return wrapper
    
    def trace_memory_usage(self, func):
        """Decorator for GPU memory monitoring (RTX 2080 Ti specific)"""
        if not self.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span("memory_monitoring") as span:
                try:
                    result = func(*args, **kwargs)
                    
                    if isinstance(result, dict):
                        span.set_attributes(self._create_span_attributes(
                            operation_type='memory_monitoring',
                            gpu_model='RTX_2080_Ti',
                            gpu_memory_used_mb=result.get('gpu_used_mb', 0),
                            gpu_memory_total_mb=result.get('gpu_total_mb', 11264),  # RTX 2080 Ti
                            gpu_utilization_percent=result.get('gpu_utilization_percent', 0),
                            cpu_memory_used_mb=result.get('cpu_used_mb', 0),
                            memory_alert=result.get('alert', None)
                        ))
                        
                        # Memory alerts
                        gpu_used = result.get('gpu_used_mb', 0)
                        if gpu_used > 9000:  # >9GB on RTX 2080 Ti
                            span.set_attribute('performance_alert', 'approaching_gpu_limit')
                        elif gpu_used > 10000:  # >10GB
                            span.set_attribute('performance_alert', 'critical_gpu_memory')
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    
    def _calculate_response_quality(self, response_text: str) -> float:
        """Calculate a basic response quality score for tax responses"""
        if not response_text:
            return 0.0
        
        score = 0.0
        
        # Length score (optimal range: 50-300 words)
        word_count = len(response_text.split())
        if 50 <= word_count <= 300:
            score += 0.3
        elif word_count < 20 or word_count > 500:
            score += 0.1
        else:
            score += 0.2
        
        # Citation score
        if any(citation in response_text.lower() for citation in ['pub29', 'irs', 'ftb']):
            score += 0.2
        
        # Disclaimer score
        if any(disclaimer in response_text.lower() for disclaimer in ['consult', 'professional', 'may vary']):
            score += 0.2
        
        # Coherence score (basic check for complete sentences)
        sentences = response_text.split('.')
        if len(sentences) >= 2 and all(len(s.strip()) > 5 for s in sentences[:3]):
            score += 0.3
        
        return min(score, 1.0)
    
    def create_evaluation_span(self, evaluation_type: str, **kwargs):
        """Create a span for evaluation operations"""
        if not self.enabled:
            return None
        
        span = self.tracer.start_span(f"evaluation_{evaluation_type}")
        span.set_attributes(self._create_span_attributes(
            operation_type='evaluation',
            evaluation_type=evaluation_type,
            **kwargs
        ))
        return span
    
    def log_tax_accuracy_metrics(self, accuracy_score: float, jurisdiction_accuracy: float, 
                                citation_accuracy: float, **kwargs):
        """Log tax-specific accuracy metrics to Phoenix"""
        if not self.enabled:
            return
        
        with self.tracer.start_as_current_span("tax_accuracy_evaluation") as span:
            span.set_attributes(self._create_span_attributes(
                operation_type='tax_accuracy_evaluation',
                overall_accuracy=accuracy_score,
                jurisdiction_accuracy=jurisdiction_accuracy,
                citation_accuracy=citation_accuracy,
                evaluation_timestamp=datetime.now().isoformat(),
                **kwargs
            ))
    
    def shutdown(self):
        """Cleanup Phoenix resources"""
        if self.phoenix_session:
            try:
                self.phoenix_session.close()
                logger.info("Phoenix session closed")
            except Exception as e:
                logger.warning(f"Error closing Phoenix session: {e}")


# Global tracer instance - will be initialized by config loader
phoenix_tracer: Optional[TaxRagTracer] = None


def initialize_phoenix_tracer(config: Dict[str, Any]) -> TaxRagTracer:
    """Initialize the global Phoenix tracer with configuration"""
    global phoenix_tracer
    phoenix_tracer = TaxRagTracer(config)
    return phoenix_tracer


def get_phoenix_tracer() -> Optional[TaxRagTracer]:
    """Get the global Phoenix tracer instance"""
    return phoenix_tracer