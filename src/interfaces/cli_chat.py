"""
Command Line Interface for RAG Tax System
Provides interactive chat interface for tax queries
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

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
    print(f"❌ Import error: {str(e)}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


logger = logging.getLogger(__name__)


class ChatHistory:
    """Manages chat history for CLI interface"""
    
    def __init__(self, history_file: Optional[str] = None, max_entries: int = 100):
        """Initialize chat history"""
        self.history_file = Path(history_file) if history_file else Path("./data/.chat_history")
        self.max_entries = max_entries
        self.history: List[Dict[str, Any]] = []
        self.load_history()
    
    def add_entry(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add entry to chat history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        
        self.history.append(entry)
        
        # Maintain max entries limit
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
        
        self.save_history()
    
    def load_history(self) -> None:
        """Load chat history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} chat history entries")
            except Exception as e:
                logger.warning(f"Failed to load chat history: {str(e)}")
                self.history = []
        else:
            # Create directory if it doesn't exist
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_history(self) -> None:
        """Save chat history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save chat history: {str(e)}")
    
    def get_recent_context(self, num_entries: int = 3) -> str:
        """Get recent chat context for continuity"""
        if not self.history or num_entries <= 0:
            return ""
        
        recent_entries = self.history[-num_entries:]
        context_parts = []
        
        for entry in recent_entries:
            context_parts.append(f"Previous Q: {entry['query']}")
            context_parts.append(f"Previous A: {entry['response'][:200]}...")
        
        return "\n".join(context_parts)


class CLIChat:
    """Command Line Interface for RAG Tax System"""
    
    def __init__(self):
        """Initialize CLI chat interface"""
        self.chat_history = ChatHistory()
        self.running = False
        
        # Initialize components
        self.system_initialized = False
        
        # Initialize Phoenix monitoring
        self.phoenix_tracer = None
        self.phoenix_enabled = False
        self._initialize_phoenix_monitoring()
        
        print("🏛️  RAG Tax Assistant")
        print("=" * 50)
    
    def _initialize_phoenix_monitoring(self) -> None:
        """Initialize Phoenix monitoring system"""
        try:
            self.phoenix_tracer = get_phoenix_tracer()
            if self.phoenix_tracer is None:
                self.phoenix_tracer = initialize_phoenix_tracer(config_manager.config)
            
            if self.phoenix_tracer and self.phoenix_tracer.enabled:
                self.phoenix_enabled = True
                print("🔍 Phoenix monitoring enabled - Dashboard: http://localhost:6006")
            else:
                print("⚠️  Phoenix monitoring disabled")
        except Exception as e:
            print(f"⚠️  Phoenix monitoring unavailable: {str(e)}")
            self.phoenix_enabled = False
    
    def initialize_system(self) -> bool:
        """Initialize RAG system components"""
        if self.system_initialized:
            return True
        
        try:
            print("🔧 Initializing RAG Tax System...")
            
            # Check if vector database has documents
            stats = vector_store.get_collection_stats()
            
            if stats.get("document_count", 0) == 0:
                print("📄 No documents found in vector store. Processing PDF...")
                self.process_documents()
            else:
                print(f"📚 Found {stats['document_count']} documents in vector store")
            
            # Load models
            print("🤖 Loading embedding model...")
            embedding_service.load_model()
            
            print("🧠 Loading language model...")
            llm_service.load_model()
            
            print("✅ System initialized successfully!")
            print(f"💾 {memory_monitor.get_memory_summary()}")
            
            self.system_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize system: {str(e)}")
            logger.error(f"System initialization failed: {str(e)}")
            return False
    
    def process_documents(self) -> None:
        """Process PDF documents and populate vector store"""
        try:
            # Process PDF
            chunks = document_processor.process_pdf()
            
            if not chunks:
                print("❌ No chunks extracted from PDF")
                return
            
            print(f"📝 Extracted {len(chunks)} chunks from PDF")
            
            # Generate embeddings
            print("🔗 Generating embeddings...")
            documents = [chunk.content for chunk in chunks]
            embeddings = embedding_service.embed_documents(documents)
            
            # Store in vector database
            print("💾 Storing in vector database...")
            metadata = [chunk.metadata for chunk in chunks]
            vector_store.add_documents(documents, embeddings, metadata)
            
            print("✅ Documents processed and stored successfully!")
            
        except Exception as e:
            print(f"❌ Failed to process documents: {str(e)}")
            logger.error(f"Document processing failed: {str(e)}")
    
    def display_help(self) -> None:
        """Display help information"""
        help_text = """
🆘 Available Commands:
  
  📝 Ask any tax question (e.g., "What is B&O tax?")
  👤 Ask personal tax questions (e.g., "What are Sarah Johnson's tax obligations?")
  
  📊 System Commands:
  /help     - Show this help message
  /history  - Show recent chat history
  /stats    - Show system statistics
  /memory   - Show memory usage
  /users    - List all users in database
  /dbstatus - Check database connection status
  /clear    - Clear the screen
  /reset    - Reset vector database
  /exit     - Exit the application
  
  🔍 Phoenix Monitoring Commands:
  /phoenix  - Show Phoenix dashboard info
  /evaluate - Run comprehensive evaluation
  /metrics  - Show current performance metrics
  /alerts   - Show recent alerts and warnings
  /trends   - Show performance trends (7 days)
  /quality  - Show quality assessment summary
  
💡 Tips:
  • Ask specific questions about Washington State business taxes
  • Reference specific sections or topics for better results
  • For personal queries, mention user names, emails, or tax IDs
  • The system will automatically detect and route your queries
  • Use Phoenix monitoring commands to track system performance
"""
        print(help_text)
    
    def display_history(self) -> None:
        """Display recent chat history"""
        if not self.chat_history.history:
            print("📝 No chat history available")
            return
        
        print("\n📜 Recent Chat History:")
        print("-" * 50)
        
        recent_entries = self.chat_history.history[-5:]  # Show last 5 entries
        
        for i, entry in enumerate(recent_entries, 1):
            timestamp = entry["timestamp"][:19]  # Remove microseconds
            query = entry["query"][:100] + "..." if len(entry["query"]) > 100 else entry["query"]
            
            print(f"{i}. [{timestamp}]")
            print(f"   Q: {query}")
            print(f"   A: {entry['response'][:150]}...")
            print()
    
    def display_stats(self) -> None:
        """Display system statistics"""
        try:
            # Vector store stats
            vs_stats = vector_store.get_collection_stats()
            
            # Model info
            embedding_info = embedding_service.get_model_info()
            llm_info = llm_service.get_model_info()
            
            print("\n📊 System Statistics:")
            print("-" * 50)
            print(f"📚 Documents in vector store: {vs_stats.get('document_count', 0)}")
            print(f"🔗 Embedding model: {embedding_info.get('model_name', 'Not loaded')}")
            print(f"🧠 LLM model: {llm_info.get('model_name', 'Not loaded')}")
            print(f"💬 Chat history entries: {len(self.chat_history.history)}")
            print(f"🔢 Queries processed: {llm_info.get('query_count', 0)}")
            print(f"🔄 Smart fallbacks triggered: {llm_info.get('fallback_count', 0)}")
            
            # Calculate fallback rate if we have queries
            query_count = llm_info.get('query_count', 0)
            fallback_count = llm_info.get('fallback_count', 0)
            if query_count > 0:
                fallback_rate = (fallback_count / query_count) * 100
                print(f"📈 Fallback rate: {fallback_rate:.1f}%")
            
        except Exception as e:
            print(f"❌ Failed to get statistics: {str(e)}")
    
    def display_memory(self) -> None:
        """Display memory usage"""
        print(f"\n🧠 {memory_monitor.get_memory_summary()}")
        memory_monitor.log_memory_stats()
    
    def process_query(self, query: str) -> None:
        """Process user query and display response using router-based flow"""
        try:
            print("🤔 Processing query...")
            
            # Query using router-based system
            result = llm_service.query_with_router(query)
            
            # Display routing information
            query_type = result.get('query_type', 'unknown')
            confidence = result.get('confidence', 0)
            
            # Check if fallback was triggered
            if result.get('fallback_triggered', False):
                print(f"🔄 Smart Fallback: {result.get('fallback_reason', 'Unknown reason')}")
                print(f"📖 Answering as general tax query instead...")
                
            elif query_type == 'general_search_with_context':
                print(f"🔍 Intelligent Routing: Context-Enhanced Search")
                
                # Show routing path
                routing_result = result.get('routing_result', {})
                routing_path = routing_result.get('routing_path', [])
                print(f"🛤️  Path: {' → '.join(routing_path)}")
                
                # Show user context
                user_name = result.get('user_name', 'Unknown')
                print(f"👤 Enhanced with: {user_name}'s personal tax data")
                
                # Show routing summary for debugging
                if routing_result.get('routing_summary'):
                    print(f"📋 Summary: {routing_result['routing_summary']}")
                    
            elif query_type == 'general_search':
                print(f"📖 Standard Routing: General Search")
                
            else:
                print(f"📖 Fallback: Unknown query type ({query_type})")
            
            print(f"\n🤖 {result['response']}")
            
            # Show sources if available
            if result.get('retrieved_documents'):
                print(f"\n📚 Sources ({result['num_retrieved']} documents found):")
                for i, (doc, similarity) in enumerate(zip(result['retrieved_documents'][:3], 
                                                         result.get('similarities', [])[:3]), 1):
                    preview = doc[:100] + "..." if len(doc) > 100 else doc
                    print(f"  {i}. ({similarity:.2f}) {preview}")
            
            # Add to history with enhanced metadata
            metadata = {
                'query_type': query_type,
                'sources_count': result.get('num_retrieved', 0),
                'similarities': result.get('similarities', [])[:3]
            }
            
            # Add fallback information
            if result.get('fallback_triggered', False):
                metadata.update({
                    'fallback_triggered': True,
                    'original_classification': result.get('original_classification', 'unknown'),
                    'fallback_reason': result.get('fallback_reason', 'Unknown'),
                    'original_error': result.get('original_error', 'Unknown')
                })
            
            # Add routing details
            if result.get('routing_result'):
                routing = result['routing_result']
                metadata.update({
                    'routing_confidence': routing.get('confidence', 0),
                    'user_identifier': routing.get('user_identifier'),
                    'requires_user_data': routing.get('requires_user_data', False)
                })
            
            self.chat_history.add_entry(
                query=query,
                response=result['response'],
                metadata=metadata
            )
            
        except Exception as e:
            print(f"❌ Failed to process query: {str(e)}")
            logger.error(f"Query processing failed: {str(e)}")
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == "/help":
            self.display_help()
        elif command == "/history":
            self.display_history()
        elif command == "/stats":
            self.display_stats()
        elif command == "/memory":
            self.display_memory()
        elif command == "/users":
            self.display_users()
        elif command == "/dbstatus":
            self.display_db_status()
        elif command == "/clear":
            os.system('clear' if os.name == 'posix' else 'cls')
        elif command == "/reset":
            try:
                vector_store.reset_collection()
                print("🗑️  Vector database reset successfully")
            except Exception as e:
                print(f"❌ Failed to reset database: {str(e)}")
        elif command == "/phoenix":
            self.display_phoenix_info()
        elif command == "/evaluate":
            self.run_evaluation()
        elif command == "/metrics":
            self.display_metrics()
        elif command == "/alerts":
            self.display_alerts()
        elif command == "/trends":
            self.display_trends()
        elif command == "/quality":
            self.display_quality_summary()
        elif command == "/exit":
            return True
        else:
            print("❓ Unknown command. Type /help for available commands.")
        
        return False
    
    def display_users(self) -> None:
        """Display all users in the database"""
        try:
            from src.core.tax_mcp_client import mcp_server
            users = mcp_server.get_all_users()
            
            if not users:
                print("👥 No users found in database")
                return
            
            print(f"\n👥 Database Users ({len(users)} total):")
            print("-" * 60)
            
            for user in users:
                print(f"📧 {user.first_name} {user.last_name}")
                print(f"   Email: {user.email}")
                print(f"   Filing Status: {user.filing_status}")
                print(f"   Income: ${user.annual_income:,.2f}")
                print(f"   State: {user.state}")
                print()
                
        except Exception as e:
            print(f"❌ Failed to retrieve users: {str(e)}")
    
    def display_db_status(self) -> None:
        """Display database connection status"""
        try:
            from src.core.tax_mcp_client import mcp_server
            status = mcp_server.check_database_connection()
            
            print(f"\n💾 Database Status:")
            print("-" * 40)
            
            if status['status'] == 'connected':
                print(f"✅ Status: {status['status'].upper()}")
                print(f"📂 Path: {status['database_path']}")
                print(f"👥 Users: {status['user_count']}")
                print(f"💬 Message: {status['message']}")
            else:
                print(f"❌ Status: {status['status'].upper()}")
                print(f"💬 Message: {status['message']}")
                
        except Exception as e:
            print(f"❌ Failed to check database status: {str(e)}")
    
    def run(self) -> None:
        """Run the CLI chat interface"""
        self.running = True
        
        # Initialize system
        if not self.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Display welcome message
        print("\n✨ RAG Tax Assistant is ready!")
        print("💬 Ask me anything about Washington State business taxes")
        print("❓ Type /help for available commands or /exit to quit\n")
        
        try:
            while self.running:
                # Get user input
                try:
                    user_input = input("🏛️  You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if self.handle_command(user_input):
                        break
                    continue
                
                # Process query
                self.process_query(user_input)
                print()  # Add spacing
                
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            logger.error(f"CLI runtime error: {str(e)}")
        finally:
            print("🔄 Cleaning up...")
            self.cleanup()
    
    def display_phoenix_info(self) -> None:
        """Display Phoenix monitoring information"""
        print("\n🔍 Phoenix Monitoring Status:")
        print("-" * 50)
        
        if self.phoenix_enabled:
            try:
                dashboard = TaxRagDashboard()
                dashboard_url = dashboard.get_dashboard_url()
                print(f"✅ Status: ENABLED")
                print(f"🌐 Dashboard URL: {dashboard_url}")
                print(f"📊 Tracing: Active")
                print(f"🔔 Alerts: Configured")
                
                # Show current configuration
                dashboard = TaxRagDashboard()
                config = dashboard.setup_dashboard()
                if config.get("status") == "success":
                    print(f"📈 Dashboard Views: {len(config['config']['views'])}")
                    print(f"🚨 Alert Rules: {len(config['config']['alerts'])}")
                    print(f"📏 Custom Metrics: {len(config['config']['metrics'])}")
                
                print(f"\n💡 Tip: Visit {dashboard_url} to view real-time metrics")
                
            except Exception as e:
                print(f"⚠️  Phoenix enabled but dashboard unavailable: {str(e)}")
        else:
            print("❌ Status: DISABLED")
            print("💡 Phoenix monitoring is not available")
    
    def run_evaluation(self) -> None:
        """Run comprehensive evaluation"""
        if not self.phoenix_enabled:
            print("⚠️  Phoenix monitoring required for evaluations")
            return
        
        print("\n🧪 Running Comprehensive Evaluation...")
        print("-" * 50)
        print("⏳ This may take 2-3 minutes to complete...")
        
        try:
            result = automated_evaluation_service.run_comprehensive_evaluation()
            
            if result.get("status") == "disabled":
                print("⚠️  Automated evaluation is disabled")
                return
            
            # Display summary results
            summary = result.get("summary", {})
            print(f"\n📊 Evaluation Results:")
            print(f"   Overall Accuracy: {summary.get('overall_accuracy', 0):.3f}")
            print(f"   Performance Status: {summary.get('performance_status', 'unknown').upper()}")
            
            # Display key metrics
            metrics = summary.get("key_metrics", {})
            if metrics:
                print(f"\n📈 Key Metrics:")
                for metric, value in metrics.items():
                    print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
            
            # Display alerts
            alerts = result.get("alerts", [])
            if alerts:
                print(f"\n🚨 Alerts ({len(alerts)}):")
                for alert in alerts:
                    severity_emoji = {"low": "💡", "medium": "⚠️", "high": "🔥", "critical": "🚨"}
                    emoji = severity_emoji.get(alert.get("severity", "medium"), "⚠️")
                    print(f"   {emoji} {alert.get('message', 'Unknown alert')}")
            else:
                print(f"\n✅ No alerts - System performing well!")
            
            # Display recommendations
            recommendations = summary.get("recommendations", [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print(f"\n📁 Detailed report: /data/evaluation_reports/{result.get('evaluation_id', 'unknown')}.json")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
    
    def display_metrics(self) -> None:
        """Display current performance metrics"""
        print("\n📊 Current Performance Metrics:")
        print("-" * 50)
        
        try:
            # System metrics
            vs_stats = vector_store.get_collection_stats()
            llm_info = llm_service.get_model_info()
            memory_stats = memory_monitor.get_memory_stats()
            
            print("🖥️  System Performance:")
            print(f"   Documents in Vector Store: {vs_stats.get('document_count', 0)}")
            print(f"   Queries Processed: {llm_info.get('query_count', 0)}")
            print(f"   GPU Memory Used: {memory_stats.get('gpu_used_mb', 0):.0f}MB")
            print(f"   GPU Utilization: {memory_stats.get('gpu_utilization_percent', 0):.1f}%")
            
            # Recent performance summary
            if self.phoenix_enabled:
                try:
                    perf_summary = automated_evaluation_service.get_recent_performance_summary(days=1)
                    if perf_summary.get("status") == "success":
                        print(f"\n🎯 Recent Performance (24h):")
                        print(f"   Evaluations Run: {perf_summary.get('num_evaluations', 0)}")
                        print(f"   Average Accuracy: {perf_summary.get('avg_overall_accuracy', 0):.3f}")
                        print(f"   Total Alerts: {perf_summary.get('total_alerts', 0)}")
                        print(f"   Trend: {perf_summary.get('performance_trend', 'unknown').upper()}")
                except Exception as e:
                    print(f"⚠️  Recent performance data unavailable: {str(e)}")
            
        except Exception as e:
            print(f"❌ Failed to get metrics: {str(e)}")
    
    def display_alerts(self) -> None:
        """Display recent alerts and warnings"""
        print("\n🚨 Recent Alerts and Warnings:")
        print("-" * 50)
        
        if not self.phoenix_enabled:
            print("⚠️  Phoenix monitoring required for alert history")
            return
        
        try:
            # Get recent evaluation results to check for alerts
            recent_evals = automated_evaluation_service.evaluation_history[-5:]  # Last 5 evaluations
            
            if not recent_evals:
                print("📊 No recent evaluations available")
                return
            
            total_alerts = 0
            for eval_result in recent_evals:
                alerts = eval_result.get("alerts", [])
                if alerts:
                    eval_id = eval_result.get("evaluation_id", "unknown")
                    timestamp = eval_result.get("timestamp", "unknown")[:19]
                    
                    print(f"\n📅 Evaluation: {eval_id} ({timestamp})")
                    for alert in alerts:
                        severity = alert.get("severity", "medium")
                        message = alert.get("message", "Unknown alert")
                        severity_emoji = {"low": "💡", "medium": "⚠️", "high": "🔥", "critical": "🚨"}
                        emoji = severity_emoji.get(severity, "⚠️")
                        
                        print(f"   {emoji} [{severity.upper()}] {message}")
                        total_alerts += 1
            
            if total_alerts == 0:
                print("✅ No alerts in recent evaluations - System healthy!")
            else:
                print(f"\n📊 Total alerts found: {total_alerts}")
                
        except Exception as e:
            print(f"❌ Failed to get alerts: {str(e)}")
    
    def display_trends(self) -> None:
        """Display performance trends over the last 7 days"""
        print("\n📈 Performance Trends (7 days):")
        print("-" * 50)
        
        if not self.phoenix_enabled:
            print("⚠️  Phoenix monitoring required for trend analysis")
            return
        
        try:
            perf_summary = automated_evaluation_service.get_recent_performance_summary(days=7)
            
            if perf_summary.get("status") != "success":
                print("📊 No trend data available")
                return
            
            print(f"📅 Period: Last 7 days")
            print(f"📊 Evaluations: {perf_summary.get('num_evaluations', 0)}")
            print(f"🎯 Average Accuracy: {perf_summary.get('avg_overall_accuracy', 0):.3f}")
            print(f"🚨 Total Alerts: {perf_summary.get('total_alerts', 0)}")
            print(f"📈 Trend: {perf_summary.get('performance_trend', 'unknown').upper()}")
            
            # Performance trends from evaluation service
            trends = automated_evaluation_service.performance_trends
            if trends:
                print(f"\n📊 Metric Trends:")
                for metric, data_points in trends.items():
                    if data_points and len(data_points) >= 2:
                        recent_avg = sum(p['value'] for p in data_points[-3:]) / min(3, len(data_points))
                        older_avg = sum(p['value'] for p in data_points[-7:-3]) / max(1, len(data_points[-7:-3]))
                        
                        if older_avg > 0:
                            change_pct = ((recent_avg - older_avg) / older_avg) * 100
                            trend_emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                            print(f"   {metric.replace('_', ' ').title()}: {recent_avg:.3f} {trend_emoji} ({change_pct:+.1f}%)")
            
        except Exception as e:
            print(f"❌ Failed to get trends: {str(e)}")
    
    def display_quality_summary(self) -> None:
        """Display quality assessment summary"""
        print("\n⭐ Quality Assessment Summary:")
        print("-" * 50)
        
        if not self.phoenix_enabled:
            print("⚠️  Phoenix monitoring required for quality assessment")
            return
        
        try:
            # Get most recent evaluation
            recent_evals = automated_evaluation_service.evaluation_history[-1:] if automated_evaluation_service.evaluation_history else []
            
            if not recent_evals:
                print("🧪 Run an evaluation first using /evaluate")
                return
            
            latest_eval = recent_evals[0]
            results = latest_eval.get("results", {})
            summary = latest_eval.get("summary", {})
            
            # Overall quality score
            overall_accuracy = summary.get("overall_accuracy", 0)
            performance_status = summary.get("performance_status", "unknown")
            
            status_emoji = {
                "excellent": "🌟",
                "good": "✅", 
                "acceptable": "⚠️",
                "needs_improvement": "🔧"
            }
            emoji = status_emoji.get(performance_status, "❓")
            
            print(f"{emoji} Overall Quality: {overall_accuracy:.3f} ({performance_status.upper()})")
            
            # Detailed quality metrics
            if "tax_accuracy" in results:
                tax_results = results["tax_accuracy"]
                print(f"\n📚 Tax Domain Quality:")
                print(f"   Tax Accuracy: {tax_results.get('avg_accuracy', 0):.3f}")
                print(f"   Jurisdiction Accuracy: {tax_results.get('avg_jurisdiction_accuracy', 0):.3f}")
                print(f"   Citation Quality: {tax_results.get('avg_citation_accuracy', 0):.3f}")
                print(f"   Tax Compliance: {tax_results.get('avg_compliance', 0):.3f}")
                print(f"   Test Cases: {tax_results.get('num_test_cases', 0)}")
            
            if "hallucination" in results:
                hall_results = results["hallucination"]
                print(f"\n🛡️  Safety & Reliability:")
                print(f"   Hallucination Risk: {hall_results.get('avg_risk', 0):.3f}")
                print(f"   High Risk Cases: {hall_results.get('high_risk_count', 0)}")
                print(f"   Safety Test Cases: {hall_results.get('num_test_cases', 0)}")
            
            if "jurisdiction" in results:
                juris_results = results["jurisdiction"]
                print(f"\n🗺️  Query Routing:")
                print(f"   Classification Accuracy: {juris_results.get('overall_accuracy', 0):.3f}")
                print(f"   Test Cases: {juris_results.get('total_predictions', 0)}")
            
            # Quality issues
            if "tax_accuracy" in results and results["tax_accuracy"].get("quality_issues"):
                issues = results["tax_accuracy"]["quality_issues"]
                print(f"\n🔍 Quality Issues Found:")
                for issue, count in issues.items():
                    print(f"   {issue.replace('_', ' ').title()}: {count}")
            
            print(f"\n📅 Last Updated: {latest_eval.get('timestamp', 'unknown')[:19]}")
            
        except Exception as e:
            print(f"❌ Failed to get quality summary: {str(e)}")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Clear GPU memory
            memory_monitor.clear_gpu_cache()
            print("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {str(e)}")


def main():
    """Main entry point for CLI"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run CLI
    cli = CLIChat()
    cli.run()


if __name__ == "__main__":
    main()