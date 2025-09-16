#!/usr/bin/env python3
"""
Basic Tax Chatbot Query Examples

This script demonstrates basic tax queries that don't require personal information.
These queries showcase the system's ability to handle general tax questions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm_service import llm_service


def run_basic_queries():
    """Run a series of basic tax queries to demonstrate system capabilities."""
    
    print("🚀 Tax Chatbot - Basic Query Examples")
    print("=" * 50)
    
    # Load the LLM service
    print("Loading models...")
    llm_service.load_model()
    print("✅ Models loaded successfully\n")
    
    # Define example queries
    queries = [
        {
            "query": "What is B&O tax and how is it calculated?",
            "description": "General business tax question"
        },
        {
            "query": "When are quarterly tax returns due?",
            "description": "Tax deadline inquiry"
        },
        {
            "query": "What are the penalties for late filing?",
            "description": "Penalty information request"
        },
        {
            "query": "How do I register my business with the Department of Revenue?",
            "description": "Business registration process"
        }
    ]
    
    # Process each query
    for i, example in enumerate(queries, 1):
        print(f"Example {i}: {example['description']}")
        print(f"Query: {example['query']}")
        print("-" * 40)
        
        try:
            # Use the router-based query method
            result = llm_service.query_with_router(example['query'])
            
            print(f"Response Type: {result.get('query_type', 'general_tax')}")
            print(f"Documents Retrieved: {result.get('num_retrieved', 'N/A')}")
            print(f"Response:\n{result['response']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "=" * 50 + "\n")
    
    print("✅ Basic query examples completed!")


if __name__ == "__main__":
    run_basic_queries()