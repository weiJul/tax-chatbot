#!/usr/bin/env python3
"""
Test Script for Fallback Behavior
Tests the fallback from jurisdiction-specific to general retrieval
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
from src.core.llama_retrieval import hierarchical_retrieval
from src.utils.jurisdiction_detector import jurisdiction_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_fallback_scenarios():
    """Test various fallback scenarios"""
    logger.info("🧪 Testing Fallback Behavior")
    
    try:
        # Initialize system
        hierarchical_retrieval.initialize()
        
        # Test scenarios
        test_cases = [
            {
                "name": "CA query with CA data available",
                "query": "California tax rates",
                "jurisdiction": "california",
                "expected_strategy": "california_specific"
            },
            {
                "name": "CA query but no relevant CA results (should fallback)",
                "query": "California fishing license tax",  # Unlikely to be in tax docs
                "jurisdiction": "california", 
                "expected_fallback": True
            },
            {
                "name": "General query (no jurisdiction)",
                "query": "What is business tax?",
                "jurisdiction": None,
                "expected_strategy": "general_fallback"
            },
            {
                "name": "Auto-detect CA jurisdiction",
                "query": "What does the FTB require for filing?",
                "jurisdiction": None,  # Let it auto-detect
                "expected_auto_detect": "california"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            logger.info(f"\n--- Test Case {i}: {case['name']} ---")
            logger.info(f"Query: '{case['query']}'")
            logger.info(f"Jurisdiction hint: {case.get('jurisdiction', 'auto-detect')}")
            
            try:
                if case['jurisdiction']:
                    result = hierarchical_retrieval.hierarchical_query(
                        case['query'], 
                        case['jurisdiction']
                    )
                else:
                    result = hierarchical_retrieval.smart_query(case['query'])
                
                # Analyze results
                strategy = result.get('retrieval_strategy', 'unknown')
                fallback_used = result.get('fallback_used', False)
                jurisdiction_success = result.get('jurisdiction_success', False)
                auto_detected = result.get('auto_detected_jurisdiction')
                num_sources = len(result.get('source_nodes', []))
                
                print(f"  ✅ Strategy Used: {strategy}")
                print(f"  📍 Jurisdiction Success: {jurisdiction_success}")
                print(f"  🔄 Fallback Used: {fallback_used}")
                print(f"  🎯 Auto-detected: {auto_detected}")
                print(f"  📚 Sources Found: {num_sources}")
                
                # Check expectations
                if 'expected_strategy' in case:
                    if strategy == case['expected_strategy']:
                        print(f"  ✅ Expected strategy achieved!")
                    else:
                        print(f"  ⚠️  Expected {case['expected_strategy']}, got {strategy}")
                
                if 'expected_fallback' in case and case['expected_fallback']:
                    if fallback_used:
                        print(f"  ✅ Fallback behavior working correctly!")
                    else:
                        print(f"  ⚠️  Expected fallback, but didn't occur")
                
                if 'expected_auto_detect' in case:
                    if auto_detected == case['expected_auto_detect']:
                        print(f"  ✅ Auto-detection working correctly!")
                    else:
                        print(f"  ⚠️  Expected {case['expected_auto_detect']}, got {auto_detected}")
                
                if result.get('response'):
                    print(f"  💬 Response: {result['response'][:150]}...")
                else:
                    print(f"  ❌ No response generated")
                    
            except Exception as e:
                logger.error(f"  ❌ Test case failed: {str(e)}")
            
            print()
        
        # Test jurisdiction data availability
        logger.info("--- Jurisdiction Data Check ---")
        for jurisdiction in ["california", "general"]:
            check = hierarchical_retrieval.check_jurisdiction_data(jurisdiction)
            print(f"{jurisdiction.capitalize()}: {check}")
        
        logger.info("\n🎉 Fallback testing complete!")
        
    except Exception as e:
        logger.error(f"Fallback testing failed: {str(e)}")
        raise


def test_jurisdiction_keywords():
    """Test jurisdiction keyword detection"""
    logger.info("\n🔍 Testing Jurisdiction Keyword Detection")
    
    test_queries = [
        # Should detect California
        "What are CA tax rates?",
        "California franchise tax information", 
        "Tell me about pub29",
        "FTB requirements for business",
        "California standard deduction",
        
        # Should be general
        "What is B&O tax?",
        "Business tax deadlines",
        "How to file taxes?",
        "Tax penalty calculations",
        
        # Edge cases
        "I live in California but need general tax info",
        "California companies filing federal taxes",
    ]
    
    for query in test_queries:
        detected = jurisdiction_detector.detect_jurisdiction(query)
        confidence = jurisdiction_detector.get_jurisdiction_confidence(query)
        terms = jurisdiction_detector.extract_jurisdiction_terms(query)
        
        print(f"Query: '{query}'")
        print(f"  → Detected: {detected or 'general'}")
        print(f"  → CA Confidence: {confidence['california']:.2f}")
        print(f"  → Found terms: {terms[:3]}")  # Show first 3 terms
        print()


if __name__ == "__main__":
    try:
        test_jurisdiction_keywords()
        test_fallback_scenarios()
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise