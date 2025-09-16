#!/usr/bin/env python3
"""
Jurisdiction-Specific Query Examples

This script demonstrates the system's jurisdiction detection capabilities,
showing how it automatically routes California-specific vs general tax queries
to the appropriate document collections.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm_service import llm_service
from src.utils.jurisdiction_detector import jurisdiction_detector
from src.core.llama_retrieval import hierarchical_retrieval


def show_system_stats():
    """Display system statistics."""
    print("📊 System Statistics:")
    print("-" * 25)
    
    try:
        # Initialize retrieval system and get stats
        hierarchical_retrieval.initialize()
        stats = hierarchical_retrieval.get_retrieval_stats()
        
        print(f"✅ Total Documents: {stats.get('total_documents', 'N/A')}")
        print(f"✅ California Documents: {stats.get('california_documents', 'N/A')}")
        print(f"✅ General Documents: {stats.get('general_documents', 'N/A')}")
        print(f"✅ Query Engines Available: {stats.get('query_engines_available', 'N/A')}")
        print()
        
    except Exception as e:
        print(f"❌ Error getting system stats: {e}")
        print()


def test_jurisdiction_detection():
    """Test jurisdiction detection on sample queries."""
    print("🔍 Testing Jurisdiction Detection:")
    print("-" * 35)
    
    test_queries = [
        "What is B&O tax?",  # General
        "California property tax rates",  # California
        "FTB requirements for residents",  # California
        "When are tax deadlines?",  # General
        "pub29 California tax information",  # California
        "Business registration process"  # General
    ]
    
    for query in test_queries:
        jurisdiction = jurisdiction_detector.detect_jurisdiction(query)
        print(f"Query: {query}")
        print(f"Detected Jurisdiction: {jurisdiction}")
        print()


def run_jurisdiction_queries():
    """Run queries that demonstrate jurisdiction-specific routing."""
    
    print("🚀 Tax Chatbot - Jurisdiction Examples")
    print("=" * 50)
    
    # Show system statistics
    show_system_stats()
    
    # Test jurisdiction detection
    test_jurisdiction_detection()
    
    # Load the LLM service
    print("Loading models...")
    llm_service.load_model()
    print("✅ Models loaded successfully\n")
    
    # Define jurisdiction-specific queries
    queries = [
        {
            "query": "What is B&O tax and how is it calculated?",
            "expected_jurisdiction": "general",
            "description": "General business tax question"
        },
        {
            "query": "What are California property tax rates?",
            "expected_jurisdiction": "california",
            "description": "California-specific tax query"
        },
        {
            "query": "Tell me about FTB requirements for California residents",
            "expected_jurisdiction": "california", 
            "description": "Franchise Tax Board (CA) query"
        },
        {
            "query": "When are quarterly tax returns due nationwide?",
            "expected_jurisdiction": "general",
            "description": "General tax deadline question"
        },
        {
            "query": "What is special about tax in California?",
            "expected_jurisdiction": "california",
            "description": "Open-ended California tax query"
        }
    ]
    
    # Process each jurisdiction query
    for i, example in enumerate(queries, 1):
        print(f"Example {i}: {example['description']}")
        print(f"Query: {example['query']}")
        print(f"Expected Jurisdiction: {example['expected_jurisdiction']}")
        
        # Test jurisdiction detection
        detected_jurisdiction = jurisdiction_detector.detect_jurisdiction(example['query'])
        print(f"Detected Jurisdiction: {detected_jurisdiction}")
        
        # Check if detection matches expectation
        detection_correct = detected_jurisdiction == example['expected_jurisdiction']
        print(f"Detection Correct: {'✅' if detection_correct else '❌'}")
        
        print("-" * 40)
        
        try:
            # Use the router-based query method
            result = llm_service.query_with_router(example['query'])
            
            print(f"Response Type: {result.get('query_type', 'general_tax')}")
            print(f"Documents Retrieved: {result.get('num_retrieved', 'N/A')}")
            
            # Show a snippet of the response
            response = result['response']
            if len(response) > 300:
                response = response[:300] + "..."
            print(f"Response Preview:\n{response}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "=" * 50 + "\n")
    
    print("✅ Jurisdiction query examples completed!")


if __name__ == "__main__":
    run_jurisdiction_queries()