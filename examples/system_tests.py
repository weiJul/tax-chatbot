#!/usr/bin/env python3
"""
System Testing Examples

This script provides examples for testing various system components
and validating system performance and functionality.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import ConfigManager
from src.core.embeddings import embedding_service
from src.core.vector_store import vector_store
from src.core.llm_service import llm_service
from src.core.tax_mcp_client import mcp_server
from src.utils.memory_monitor import memory_monitor


def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing Configuration:")
    print("-" * 25)
    
    try:
        config = ConfigManager()
        print(f"✅ Configuration loaded successfully")
        print(f"✅ LLM Model: {config.llm.model_name}")
        print(f"✅ Embedding Model: {config.embedding.model_name}")
        print(f"✅ Vector Store Collection: {config.vector_store.collection_name}")
        print()
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print()


def test_embeddings():
    """Test embedding generation."""
    print("🔤 Testing Embeddings:")
    print("-" * 22)
    
    try:
        embedding_service.load_model()
        
        # Test embedding generation
        test_text = "What is business and occupation tax?"
        embedding = embedding_service.embed_text(test_text)
        
        print(f"✅ Embedding model loaded: {embedding_service.model_name}")
        print(f"✅ Embedding generated - Shape: {embedding.shape}")
        print(f"✅ Embedding dimension: {embedding.shape[0]}")
        print()
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        print()


def test_vector_store():
    """Test vector store connectivity and stats."""
    print("🗄️  Testing Vector Store:")
    print("-" * 26)
    
    try:
        vector_store.connect()
        stats = vector_store.get_collection_stats()
        
        print(f"✅ Vector store connected")
        print(f"✅ Collection: {stats['collection_name']}")
        print(f"✅ Total documents: {stats['total_documents']}")
        
        # Test search functionality
        if stats['total_documents'] > 0:
            query_embedding = embedding_service.embed_query("tax rates")
            docs, metadata, similarities = vector_store.search(query_embedding, top_k=3)
            print(f"✅ Search test: Found {len(docs)} results")
            if similarities:
                print(f"✅ Top similarity score: {similarities[0]:.3f}")
        
        print()
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        print()


def test_mcp_database():
    """Test MCP database connection and user lookup."""
    print("📊 Testing MCP Database:")
    print("-" * 26)
    
    try:
        # Test database connection
        status = mcp_server.check_database_connection()
        print(f"✅ Database status: {status['status']}")
        print(f"✅ User count: {status.get('user_count', 'N/A')}")
        
        # Test user lookup
        user = mcp_server.get_user_by_name("Sarah", "Johnson")
        if user:
            print(f"✅ User lookup successful: {user.first_name} {user.last_name}")
            print(f"✅ Email: {user.email}")
            print(f"✅ Filing status: {user.filing_status}")
        else:
            print("❌ Demo user not found")
        
        print()
    except Exception as e:
        print(f"❌ MCP database test failed: {e}")
        print()


def test_memory_monitoring():
    """Test memory monitoring functionality."""
    print("💾 Testing Memory Monitoring:")
    print("-" * 30)
    
    try:
        # Get memory statistics
        stats = memory_monitor.get_memory_stats()
        
        print(f"✅ System RAM: {stats['system']['total_gb']:.1f}GB total, "
              f"{stats['system']['available_gb']:.1f}GB available")
        
        if stats['gpu']['available']:
            for i, gpu in enumerate(stats['gpu']['devices']):
                print(f"✅ GPU {i}: {gpu['name']}")
                print(f"   Memory: {gpu['memory_used_gb']:.1f}/{gpu['memory_total_gb']:.1f}GB")
        else:
            print("ℹ️  No GPU detected (CPU-only mode)")
        
        print()
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        print()


def test_llm_performance():
    """Test LLM loading and basic performance."""
    print("🧠 Testing LLM Performance:")
    print("-" * 27)
    
    try:
        # Load LLM
        start_time = time.time()
        llm_service.load_model()
        load_time = time.time() - start_time
        
        print(f"✅ LLM loaded in {load_time:.2f} seconds")
        
        # Get model info
        info = llm_service.get_model_info()
        print(f"✅ Model: {info['model_name']}")
        print(f"✅ Status: {info['status']}")
        
        # Test basic query
        test_query = "What is tax?"
        start_time = time.time()
        result = llm_service.query_with_router(test_query)
        query_time = time.time() - start_time
        
        print(f"✅ Query processed in {query_time:.2f} seconds")
        print(f"✅ Query type: {result.get('query_type', 'unknown')}")
        print(f"✅ Documents retrieved: {result.get('num_retrieved', 'N/A')}")
        
        print()
    except Exception as e:
        print(f"❌ LLM performance test failed: {e}")
        print()


def run_system_tests():
    """Run comprehensive system tests."""
    
    print("🚀 Tax Chatbot - System Testing")
    print("=" * 50)
    
    # Run all component tests
    test_configuration()
    test_embeddings()
    test_vector_store()
    test_mcp_database()
    test_memory_monitoring()
    test_llm_performance()
    
    print("✅ All system tests completed!")
    print("\nℹ️  If any tests failed, check the installation guide")
    print("   and ensure all dependencies are properly installed.")


if __name__ == "__main__":
    run_system_tests()