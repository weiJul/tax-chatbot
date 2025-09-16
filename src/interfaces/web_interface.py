"""
Streamlit Web Interface for RAG Tax System
Provides web-based chat interface for tax queries
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import after path modification
try:
    from src.core.config import config_manager
    from src.core.document_processor import document_processor
    from src.core.embeddings import embedding_service
    from src.core.llm_service import llm_service
    from src.core.phoenix_tracer import (get_phoenix_tracer,
                                         initialize_phoenix_tracer)
    from src.core.vector_store import vector_store
    from src.evaluation.automated_evaluation import \
        automated_evaluation_service
    from src.monitoring.phoenix_dashboard import TaxRagDashboard
    from src.utils.memory_monitor import memory_monitor
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Make sure you're running from the project root directory")
    st.stop()


logger = logging.getLogger(__name__)


class RAGTaxApp:
    """Streamlit web application for RAG Tax System"""
    
    def __init__(self):
        """Initialize the web application"""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="RAG Tax Assistant",
            page_icon="🏛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = False
            st.session_state.chat_history = []
            st.session_state.system_stats = {}
            st.session_state.phoenix_enabled = False
            st.session_state.monitoring_stats = {}
            st.session_state.show_monitoring = False
            
            # Initialize Phoenix monitoring
            self.initialize_phoenix_monitoring()
    
    def initialize_phoenix_monitoring(self):
        """Initialize Phoenix monitoring for web interface"""
        try:
            phoenix_tracer = get_phoenix_tracer()
            if phoenix_tracer is None:
                phoenix_tracer = initialize_phoenix_tracer(config_manager.config)
            
            if phoenix_tracer and phoenix_tracer.enabled:
                st.session_state.phoenix_enabled = True
                # Initialize dashboard when needed
                dashboard = TaxRagDashboard()
                st.session_state.phoenix_dashboard_url = dashboard.get_dashboard_url()
            else:
                st.session_state.phoenix_enabled = False
        except Exception as e:
            logger.error(f"Phoenix monitoring initialization failed: {str(e)}")
            st.session_state.phoenix_enabled = False
    
    @staticmethod
    @st.cache_resource
    def initialize_system():
        """Initialize RAG system components (cached)"""
        try:
            # Check if vector database has documents
            vector_store.connect()
            stats = vector_store.get_collection_stats()
            
            if stats.get("document_count", 0) == 0:
                # Process documents if none exist
                chunks = document_processor.process_pdf()
                if chunks:
                    documents = [chunk.content for chunk in chunks]
                    embeddings = embedding_service.embed_documents(documents)
                    metadata = [chunk.metadata for chunk in chunks]
                    vector_store.add_documents(documents, embeddings, metadata)
            
            # Load models
            embedding_service.load_model()
            llm_service.load_model()
            
            return True, "System initialized successfully!"
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False, f"Failed to initialize system: {str(e)}"
    
    def display_header(self):
        """Display application header"""
        st.title("🏛️ RAG Tax Assistant")
        st.markdown("**Ask questions about Washington State business taxes**")
        st.divider()
    
    def display_sidebar(self):
        """Display sidebar with system information and controls"""
        with st.sidebar:
            st.header("🔧 System Status")
            
            # System initialization
            if not st.session_state.initialized:
                if st.button("🚀 Initialize System", type="primary"):
                    with st.spinner("Initializing RAG system..."):
                        success, message = self.initialize_system()
                        if success:
                            st.session_state.initialized = True
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                return
            
            st.success("✅ System Ready")
            
            # System statistics
            if st.button("📊 Refresh Stats"):
                self.update_system_stats()
            
            if st.session_state.system_stats:
                stats = st.session_state.system_stats
                st.metric("📚 Documents", stats.get("document_count", 0))
                st.metric("💬 Queries", stats.get("query_count", 0))
                st.metric("🧠 GPU Memory", f"{stats.get('gpu_memory_gb', 0):.1f} GB")
            
            # Memory management
            st.subheader("🧠 Memory Management")
            if st.button("🗑️ Clear GPU Cache"):
                memory_monitor.clear_gpu_cache()
                st.success("GPU cache cleared!")
            
            # Phoenix Monitoring
            st.subheader("🔍 Phoenix Monitoring")
            self.display_phoenix_monitoring_panel()
            
            # Database management
            st.subheader("💾 Database Management")
            self.display_database_panel()
            
            # Chat history management
            st.subheader("💬 Chat Management")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                st.subheader("RAG Parameters")
                
                top_k = st.slider("Top K Documents", 1, 10, 5)
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
                max_tokens = st.slider("Max Response Tokens", 100, 1000, 512, 50)
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                
                st.session_state.rag_params = {
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Debug mode toggle
                st.subheader("Debug Options")
                debug_mode = st.checkbox("🔧 Enable Debug Info", value=False, help="Show detailed routing information")
                st.session_state.debug_mode = debug_mode
    
    def update_system_stats(self):
        """Update system statistics"""
        try:
            vs_stats = vector_store.get_collection_stats()
            llm_info = llm_service.get_model_info()
            memory_stats = memory_monitor.get_memory_stats()
            
            st.session_state.system_stats = {
                "document_count": vs_stats.get("document_count", 0),
                "query_count": llm_info.get("query_count", 0),
                "gpu_memory_gb": memory_stats.gpu_reserved_mb / 1024 if memory_stats else 0
            }
        except Exception as e:
            logger.error(f"Failed to update stats: {str(e)}")
    
    def display_database_panel(self):
        """Display database management panel in sidebar"""
        try:
            from src.core.tax_mcp_client import mcp_server

            # Database status
            if st.button("🔍 Check Database Status"):
                status = mcp_server.check_database_connection()
                if status['status'] == 'connected':
                    st.success(f"✅ Connected - {status['user_count']} users")
                else:
                    st.error(f"❌ {status['message']}")
            
            # User browser
            if st.button("👥 Browse Users"):
                users = mcp_server.get_all_users()
                if users:
                    st.session_state.show_users = True
                    st.session_state.users_list = users
                else:
                    st.warning("No users found in database")
            
            # Display users if requested
            if getattr(st.session_state, 'show_users', False):
                users = getattr(st.session_state, 'users_list', [])
                with st.expander(f"👥 Database Users ({len(users)})", expanded=True):
                    for user in users:
                        st.write(f"**{user.first_name} {user.last_name}**")
                        st.write(f"📧 {user.email}")
                        st.write(f"💰 ${user.annual_income:,.2f} ({user.filing_status})")
                        st.write(f"🏛️ {user.state}")
                        st.divider()
                    
                    if st.button("❌ Close User List"):
                        st.session_state.show_users = False
                        st.rerun()
            
            # Test user search
            with st.expander("🔍 Test User Search"):
                search_term = st.text_input("Search for user:", placeholder="e.g., Sarah Johnson or sarah.johnson@email.com")
                if st.button("Search") and search_term:
                    # Test the extraction and search using the same path as main queries
                    test_query = f"Find information for {search_term}"
                    try:
                        # Use llm_service.query_with_router to get consistent results
                        result = llm_service.query_with_router(test_query)
                        
                        if result.get('user_context') and result['user_context'].found:
                            user_data = result['user_context'].user_data
                            st.success(f"✅ Found: {user_data.first_name} {user_data.last_name}")
                            st.json(result['user_context'].tax_context)
                        else:
                            st.error(f"❌ User not found")
                            # Show routing result details for debugging
                            if result.get('routing_result'):
                                routing_result = result['routing_result']
                                if routing_result.get('extraction_attempts'):
                                    st.write("Extraction attempts:")
                                    for attempt in routing_result['extraction_attempts']:
                                        st.write(f"- {attempt['identifier']} ({attempt['method']})")
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
            
        except Exception as e:
            st.error(f"Database panel error: {str(e)}")
    
    def display_phoenix_monitoring_panel(self):
        """Display Phoenix monitoring panel in sidebar"""
        try:
            # Phoenix Status
            phoenix_tracer = get_phoenix_tracer()
            if phoenix_tracer and phoenix_tracer.enabled:
                st.success("✅ Phoenix Monitoring Active")
                
                # Phoenix Dashboard Link
                phoenix_config = config_manager.phoenix
                dashboard_url = f"http://{phoenix_config.server.host}:{phoenix_config.server.port}"
                st.markdown(f"🌐 [Open Phoenix Dashboard]({dashboard_url})")
                
                # Quick Stats
                if st.button("📊 Refresh Phoenix Stats"):
                    try:
                        # Get basic tracing stats
                        tracer_stats = phoenix_tracer.get_session_stats()
                        if tracer_stats:
                            st.metric("🔄 Total Traces", tracer_stats.get('total_traces', 0))
                            st.metric("⚡ Active Spans", tracer_stats.get('active_spans', 0))
                            st.metric("📈 Avg Response Time", f"{tracer_stats.get('avg_response_time', 0):.2f}ms")
                    except Exception as e:
                        st.warning(f"Stats unavailable: {str(e)}")
                
                # Evaluation Controls
                st.divider()
                st.write("**🧪 Evaluation Controls**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Run Quick Eval"):
                        try:
                            with st.spinner("Running evaluation..."):
                                results = automated_evaluation_service.run_quick_evaluation()
                                if results:
                                    st.success(f"✅ Quick evaluation completed!")
                                    st.session_state.last_eval_results = results
                                else:
                                    st.warning("Evaluation completed with no results")
                        except Exception as e:
                            st.error(f"Evaluation failed: {str(e)}")
                
                with col2:
                    if st.button("📋 View Results"):
                        if hasattr(st.session_state, 'last_eval_results') and st.session_state.last_eval_results:
                            with st.expander("📊 Latest Evaluation Results", expanded=True):
                                results = st.session_state.last_eval_results
                                
                                # Display summary metrics
                                if 'summary' in results:
                                    summary = results['summary']
                                    st.metric("Tax Accuracy", f"{summary.get('tax_accuracy', 0):.1%}")
                                    st.metric("Hallucination Rate", f"{summary.get('hallucination_rate', 0):.1%}")
                                    st.metric("Jurisdiction Accuracy", f"{summary.get('jurisdiction_accuracy', 0):.1%}")
                                
                                # Display detailed results
                                if 'detailed_results' in results:
                                    st.json(results['detailed_results'])
                        else:
                            st.info("No evaluation results available. Run an evaluation first.")
                
                # Alerting Status
                st.divider()
                st.write("**🚨 Alert Status**")
                
                try:
                    dashboard = TaxRagDashboard()
                    alert_status = dashboard.get_alert_status()
                    if alert_status.get('active_alerts', 0) > 0:
                        st.error(f"⚠️ {alert_status['active_alerts']} Active Alerts")
                    else:
                        st.success("✅ No Active Alerts")
                except Exception as e:
                    st.info("Alert status unavailable")
                
                # Phoenix Configuration
                with st.expander("⚙️ Phoenix Configuration"):
                    st.write(f"**Project:** {phoenix_config.tracing.project_name}")
                    st.write(f"**Environment:** {phoenix_config.tracing.environment}")
                    st.write(f"**Server:** {phoenix_config.server.host}:{phoenix_config.server.port}")
                    st.write(f"**Auto-evaluation:** {'✅ Enabled' if phoenix_config.evaluation.enabled else '❌ Disabled'}")
                    
            else:
                st.warning("⚠️ Phoenix Monitoring Inactive")
                if st.button("🔧 Initialize Phoenix"):
                    try:
                        with st.spinner("Initializing Phoenix monitoring..."):
                            initialize_phoenix_tracer()
                            st.success("✅ Phoenix monitoring initialized!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize Phoenix: {str(e)}")
                        
        except Exception as e:
            st.error(f"Phoenix panel error: {str(e)}")
            logger.error(f"Phoenix monitoring panel error: {str(e)}")
    
    def display_chat_interface(self):
        """Display main chat interface"""
        if not st.session_state.initialized:
            st.info("👈 Please initialize the system using the sidebar")
            return
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.write(message["query"])
                
                # Assistant message
                with st.chat_message("assistant"):
                    # Display routing information
                    routing_info = message.get("routing_info", {})
                    if routing_info:
                        self.display_routing_info(routing_info)
                    
                    st.write(message["response"])
                    
                    # Show sources if available
                    if message.get("sources"):
                        with st.expander("📚 Sources"):
                            for j, (source, similarity) in enumerate(message["sources"], 1):
                                st.write(f"**{j}.** (Similarity: {similarity:.2f})")
                                st.write(source[:300] + "..." if len(source) > 300 else source)
                                st.divider()
        
        # Chat input
        if query := st.chat_input("Ask a tax question..."):
            self.process_query(query)
    
    def process_query(self, query: str):
        """Process user query"""
        try:
            # Get RAG parameters from session state
            params = getattr(st.session_state, 'rag_params', {})
            
            # Add user message to chat
            st.session_state.chat_history.append({
                "query": query,
                "response": "",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            })
            
            # Process query with loading indicator
            with st.spinner("🤔 Thinking..."):
                result = llm_service.query_with_router(
                    query=query,
                    top_k=params.get("top_k", 5),
                    similarity_threshold=params.get("similarity_threshold", 0.7)
                )
                
                # Prepare sources for display
                sources = []
                if result.get('retrieved_documents'):
                    for doc, similarity in zip(result['retrieved_documents'][:3], 
                                             result.get('similarities', [])[:3]):
                        sources.append((doc, similarity))
                
                # Update the last message with response and routing info
                st.session_state.chat_history[-1]["response"] = result['response']
                st.session_state.chat_history[-1]["sources"] = sources
                st.session_state.chat_history[-1]["routing_info"] = {
                    "query_type": result.get('query_type', 'unknown'),
                    "confidence": result.get('confidence', 0),
                    "user_context": result.get('user_context'),
                    "successful_method": result.get('successful_method'),
                    "extraction_attempts": result.get('extraction_attempts', []),
                    "error": result.get('error')
                }
            
            # Update system stats
            self.update_system_stats()
            
            # Trigger rerun to show new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to process query: {str(e)}")
            logger.error(f"Query processing failed: {str(e)}")
    
    def display_routing_info(self, routing_info: Dict[str, Any]):
        """Display routing classification information"""
        query_type = routing_info.get("query_type", "unknown")
        confidence = routing_info.get("confidence", 0)
        
        # Create routing indicator
        if query_type == "general_search_with_context":
            st.info(f"🔍 **Context-Enhanced Search**")
            
            # Show user context if found
            user_context = routing_info.get("user_context")
            if user_context and user_context.found:
                user_data = user_context.user_data
                tax_context = user_context.tax_context
                
                # User information in expandable section
                with st.expander("👤 User Information", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Name", f"{user_data.first_name} {user_data.last_name}")
                        st.metric("Email", user_data.email)
                    with col2:
                        st.metric("Filing Status", tax_context.get('filing_status', 'Unknown'))
                        st.metric("State", tax_context.get('state', 'Unknown'))
                    with col3:
                        st.metric("Annual Income", f"${tax_context.get('annual_income', 0):,.2f}")
                        st.metric("Tax Bracket", tax_context.get('tax_bracket', 'Unknown'))
                
                # Show successful method
                if routing_info.get('successful_method'):
                    st.success(f"✅ Found via: {routing_info['successful_method']}")
            
            # Show any errors
            elif routing_info.get('error'):
                st.warning(f"⚠️ {routing_info['error']}")
                
                # Show extraction attempts
                extraction_attempts = routing_info.get('extraction_attempts', [])
                if extraction_attempts:
                    with st.expander("🎯 Extraction Attempts"):
                        for i, attempt in enumerate(extraction_attempts, 1):
                            st.write(f"{i}. **{attempt['identifier']}** (method: {attempt['method']}, confidence: {attempt['confidence']:.2f})")
        
        elif query_type == "general_tax":
            st.info(f"📖 **General Tax Query** (confidence: {confidence:.2f})")
        
        else:
            st.warning(f"❓ **Unknown Query Type** (confidence: {confidence:.2f})")
        
        # Show debug information if enabled
        if getattr(st.session_state, 'debug_mode', False):
            with st.expander("🔧 Debug Information"):
                st.json({
                    "query_type": query_type,
                    "confidence": confidence,
                    "successful_method": routing_info.get('successful_method'),
                    "extraction_attempts": routing_info.get('extraction_attempts', []),
                    "error": routing_info.get('error'),
                    "routing_full": routing_info
                })
    
    def display_example_queries(self):
        """Display example queries"""
        st.subheader("💡 Example Questions")
        
        # Categorize examples
        general_queries = [
            "What is B&O tax and how is it calculated?",
            "When are quarterly tax returns due?",
            "What are the penalties for late filing?",
            "How do I register my business with the Department of Revenue?",
            "What is the difference between retail sales tax and use tax?",
            "What deductions are available for B&O tax?"
        ]
        
        personal_queries = [
            "My name is Sarah Johnson and I need my tax ID",
            "What are Michael Chen's tax obligations?",
            "Show me emily.rodriguez@email.com filing status",
            "I am Sarah Johnson, what is my tax bracket?",
            "What does Michael Chen owe in taxes?",
            "Find tax information for Emily Rodriguez"
        ]
        
        # Create tabs for different query types
        tab1, tab2 = st.tabs(["📖 General Tax", "👤 Personal Tax"])
        
        with tab1:
            st.caption("General tax information queries")
            for i, query in enumerate(general_queries):
                if st.button(f"📝 {query}", key=f"general_{i}"):
                    if st.session_state.initialized:
                        self.process_query(query)
                    else:
                        st.warning("Please initialize the system first!")
        
        with tab2:
            st.caption("Personal tax queries (uses database)")
            st.info("💡 Available users: Sarah Johnson, Michael Chen, Emily Rodriguez")
            for i, query in enumerate(personal_queries):
                if st.button(f"👤 {query}", key=f"personal_{i}"):
                    if st.session_state.initialized:
                        self.process_query(query)
                    else:
                        st.warning("Please initialize the system first!")
    
    def display_document_info(self):
        """Display information about loaded documents"""
        with st.expander("📄 Document Information"):
            try:
                stats = vector_store.get_collection_stats()
                st.write(f"**Documents loaded:** {stats.get('document_count', 0)}")
                st.write(f"**Collection:** {stats.get('collection_name', 'Unknown')}")
                
                # Show processed chunks info if available
                chunks = document_processor.load_processed_chunks()
                if chunks:
                    chunk_stats = document_processor.get_chunk_stats(chunks)
                    st.write(f"**Total chunks:** {chunk_stats['total_chunks']}")
                    st.write(f"**Average chunk length:** {chunk_stats['average_length']:.0f} characters")
                    st.write(f"**Source document:** {Path(chunk_stats['source_document']).name}")
                
            except Exception as e:
                st.error(f"Failed to load document info: {str(e)}")
    
    def run(self):
        """Run the Streamlit application"""
        try:
            self.display_header()
            self.display_sidebar()
            
            # Main content area
            col1, col2 = st.columns([3, 1])
            
            with col1:
                self.display_chat_interface()
            
            with col2:
                self.display_example_queries()
                st.divider()
                self.display_document_info()
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Streamlit app error: {str(e)}")


def main():
    """Main entry point for Streamlit app"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run app
    app = RAGTaxApp()
    app.run()


if __name__ == "__main__":
    main()