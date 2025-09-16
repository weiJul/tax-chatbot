#!/usr/bin/env python3
"""
Real-time LLM evaluation with actual model inference
This version actually processes queries through the full RAG pipeline
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import time
from src.core.llm_service import llm_service
from src.evaluation.tax_evaluators import TaxAccuracyEvaluator

def real_time_evaluation_demo():
    """Demo showing real LLM processing times"""
    
    print("🐌 REAL-TIME LLM EVALUATION (SLOW VERSION)")
    print("=" * 60)
    print("This version uses actual LLM inference - much slower!")
    print()
    
    # Initialize evaluator
    tax_evaluator = TaxAccuracyEvaluator()
    
    # Test queries
    queries = [
        "What is business and occupation tax?",
        "How do I calculate quarterly estimated taxes?",
        "What are the penalty rates for late filing?"
    ]
    
    total_start_time = time.time()
    
    for i, query in enumerate(queries, 1):
        print(f"🔍 Query {i}/{len(queries)}: {query}")
        
        # Start timing
        query_start_time = time.time()
        
        try:
            print("  ⏳ Generating response with LLM...")
            
            # THIS IS THE SLOW PART - Real LLM inference
            result = llm_service.query_with_router(query)
            response = result.get('response', '')
            
            llm_time = time.time() - query_start_time
            print(f"  🤖 LLM Response ({llm_time:.2f}s): {response[:100]}...")
            
            # Fast evaluation part
            eval_start_time = time.time()
            accuracy_result = tax_evaluator.evaluate_tax_response(
                query=query,
                response=response
            )
            eval_time = time.time() - eval_start_time
            
            total_query_time = time.time() - query_start_time
            
            print(f"  📊 Accuracy: {accuracy_result['overall_accuracy']:.3f}")
            print(f"  ⏱️  LLM Time: {llm_time:.2f}s")
            print(f"  ⏱️  Eval Time: {eval_time:.3f}s") 
            print(f"  ⏱️  Total Time: {total_query_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
        
        print()
    
    total_time = time.time() - total_start_time
    print(f"🏁 Total evaluation time: {total_time:.2f}s")
    print(f"📊 Average per query: {total_time/len(queries):.2f}s")
    print()
    print("Compare this to the fast demo version that takes ~0.05s per query!")

if __name__ == "__main__":
    try:
        real_time_evaluation_demo()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()