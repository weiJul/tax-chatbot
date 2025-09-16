#!/usr/bin/env python3
"""
Test Script for Hierarchical Retrieval with Jurisdiction Support
Tests California-specific queries vs general tax queries and fallback behavior
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
from typing import List, Dict, Any
import time

from src.core.llm_service import llm_service
from src.core.llama_retrieval import hierarchical_retrieval
from src.utils.jurisdiction_detector import jurisdiction_detector
from src.utils.memory_monitor import memory_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_jurisdiction_detection():
    """Test the jurisdiction detection utility"""
    logger.info("\n=== Testing Jurisdiction Detection ===")
    
    test_queries = [
        # California-specific queries
        "What are California tax rates for 2024?",
        "How do I file CA state taxes?",
        "What is the California standard deduction?",
        "Tell me about pub29 regulations",
        "What does the FTB require for business filing?",
        "California sales tax information",
        
        # General tax queries  
        "What is B&O tax?",
        "When are tax deadlines?",
        "How do I calculate business taxes?",
        "What are the penalty rates for late filing?",
        "General tax preparation steps",
    ]
    
    for query in test_queries:
        jurisdiction = jurisdiction_detector.detect_jurisdiction(query)
        confidence = jurisdiction_detector.get_jurisdiction_confidence(query)
        
        print(f"Query: '{query}'")
        print(f"  → Jurisdiction: {jurisdiction or 'general'}")
        print(f"  → Confidence: CA={confidence['california']:.2f}, General={confidence['general']:.2f}")
        print()


def test_hierarchical_retrieval_basic():
    """Test basic hierarchical retrieval functionality"""
    logger.info("\n=== Testing Basic Hierarchical Retrieval ===")
    
    try:
        # Initialize the system
        logger.info("Initializing hierarchical retrieval...")
        hierarchical_retrieval.initialize()
        
        # Test system stats
        stats = hierarchical_retrieval.get_retrieval_stats()
        logger.info(f"System stats: {stats}")
        
        # Test jurisdiction data availability
        for jurisdiction in ["california", "general"]:
            jurisdiction_check = hierarchical_retrieval.check_jurisdiction_data(jurisdiction)
            logger.info(f"{jurisdiction.capitalize()} data: {jurisdiction_check}")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic retrieval test failed: {str(e)}")
        return False


def test_california_specific_queries():
    """Test California-specific queries"""
    logger.info("\n=== Testing California-Specific Queries ===")
    
    california_queries = [
        "What are California tax rates?",
        "How do I file CA state taxes?", 
        "What is pub29 about?",
        "California franchise tax board requirements",
        "CA disability insurance tax rates"
    ]
    
    results = []
    
    for query in california_queries:
        logger.info(f"\nTesting CA query: '{query}'")
        
        try:
            start_time = time.time()
            
            # Test with explicit California jurisdiction
            result = hierarchical_retrieval.hierarchical_query(query, "california")
            
            end_time = time.time()
            query_time = end_time - start_time
            
            print(f"Query: '{query}'")
            print(f"Strategy: {result.get('retrieval_strategy', 'unknown')}")
            print(f"Jurisdiction Success: {result.get('jurisdiction_success', False)}")
            print(f"Fallback Used: {result.get('fallback_used', False)}")
            print(f"Sources Found: {len(result.get('source_nodes', []))}")
            print(f"Query Time: {query_time:.2f}s")
            
            if result.get('response'):
                print(f"Response Preview: {result['response'][:200]}...")
            
            results.append({
                "query": query,
                "success": bool(result.get('response')),
                "strategy": result.get('retrieval_strategy'),
                "query_time": query_time,
                "jurisdiction_success": result.get('jurisdiction_success', False)
            })
            
            print("-" * 60)
            
        except Exception as e:
            logger.error(f"Failed to process CA query '{query}': {str(e)}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful_queries = [r for r in results if r.get('success')]
    ca_successful = [r for r in results if r.get('jurisdiction_success')]
    
    logger.info(f"\nCalifornia Query Results:")
    logger.info(f"  Total queries: {len(california_queries)}")
    logger.info(f"  Successful: {len(successful_queries)}")
    logger.info(f"  CA jurisdiction success: {len(ca_successful)}")
    
    return results


def test_general_queries_with_fallback():
    """Test general queries and fallback behavior"""
    logger.info("\n=== Testing General Queries and Fallback ===")
    
    general_queries = [
        "What is B&O tax?",
        "When are business tax deadlines?",
        "How to calculate tax penalties?",
        "What are tax preparation best practices?",
        "General business tax information"
    ]
    
    results = []
    
    for query in general_queries:
        logger.info(f"\nTesting general query: '{query}'")
        
        try:
            start_time = time.time()
            
            # Test with smart auto-detection
            result = hierarchical_retrieval.smart_query(query)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            print(f"Query: '{query}'")
            print(f"Auto-detected Jurisdiction: {result.get('auto_detected_jurisdiction', 'None')}")
            print(f"Strategy: {result.get('retrieval_strategy', 'unknown')}")
            print(f"Sources Found: {len(result.get('source_nodes', []))}")
            print(f"Query Time: {query_time:.2f}s")
            
            if result.get('response'):
                print(f"Response Preview: {result['response'][:200]}...")
            
            results.append({
                "query": query,
                "success": bool(result.get('response')),
                "strategy": result.get('retrieval_strategy'),
                "auto_detected": result.get('auto_detected_jurisdiction'),
                "query_time": query_time
            })
            
            print("-" * 60)
            
        except Exception as e:
            logger.error(f"Failed to process general query '{query}': {str(e)}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful_queries = [r for r in results if r.get('success')]
    
    logger.info(f"\nGeneral Query Results:")
    logger.info(f"  Total queries: {len(general_queries)}")
    logger.info(f"  Successful: {len(successful_queries)}")
    
    return results


def test_llm_service_integration():
    """Test integration with LLM service"""
    logger.info("\n=== Testing LLM Service Integration ===")
    
    try:
        # Load LLM service (this will take some time)
        logger.info("Loading LLM service...")
        llm_service.load_model()
        
        test_queries = [
            "What are California tax rates?",  # Should use CA jurisdiction
            "What is B&O tax?",               # Should use general
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting LLM integration with: '{query}'")
            
            try:
                start_time = time.time()
                
                # Test the full hierarchical RAG pipeline
                result = llm_service.query_with_hierarchical_rag(query)
                
                end_time = time.time()
                query_time = end_time - start_time
                
                print(f"Query: '{query}'")
                print(f"Strategy: {result.get('retrieval_strategy', 'unknown')}")
                print(f"Jurisdiction: {result.get('auto_detected_jurisdiction', 'None')}")
                print(f"LLM Response: {result['response'][:300] if result.get('response') else 'No response'}...")
                print(f"Total Time: {query_time:.2f}s")
                print("-" * 60)
                
            except Exception as e:
                logger.error(f"LLM integration test failed for '{query}': {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"LLM service integration test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    logger.info("🧪 Starting Hierarchical Retrieval Tests")
    
    try:
        # Test 1: Jurisdiction Detection
        test_jurisdiction_detection()
        
        # Test 2: Basic Hierarchical Retrieval
        if not test_hierarchical_retrieval_basic():
            logger.error("Basic retrieval test failed - stopping")
            return
        
        # Test 3: California-specific queries
        ca_results = test_california_specific_queries()
        
        # Test 4: General queries and fallback
        general_results = test_general_queries_with_fallback()
        
        # Test 5: LLM Service Integration (optional - takes time)
        logger.info("\nWould you like to test LLM service integration? (This will load the full model)")
        # For automated testing, we'll skip this for now
        # test_llm_service_integration()
        
        # Final Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 TEST SUMMARY")
        logger.info("="*60)
        
        if ca_results:
            ca_success_rate = len([r for r in ca_results if r.get('success')]) / len(ca_results) * 100
            logger.info(f"California queries success rate: {ca_success_rate:.1f}%")
            
        if general_results:
            gen_success_rate = len([r for r in general_results if r.get('success')]) / len(general_results) * 100
            logger.info(f"General queries success rate: {gen_success_rate:.1f}%")
        
        logger.info("Hierarchical retrieval testing complete!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise
    
    finally:
        # Clean up
        memory_monitor.clear_gpu_cache()


if __name__ == "__main__":
    main()