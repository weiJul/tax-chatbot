#!/usr/bin/env python3
"""
Real-time evaluation of the actual tax RAG chatbot with Phoenix tracing
This script processes queries through the full RAG pipeline and evaluates the results
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any

# Set environment variables for Phoenix
os.environ['PHOENIX_HOST'] = 'localhost'
os.environ['PHOENIX_PORT'] = '6006'
os.environ['PHOENIX_GRPC_PORT'] = '6007'  # Use different port for gRPC to avoid conflicts

# Phoenix and OpenTelemetry imports
import phoenix as px
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Tax RAG system imports
from src.core.llm_service import llm_service
from src.evaluation.tax_evaluators import TaxAccuracyEvaluator, HallucinationDetector, JurisdictionClassificationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTaxChatbotEvaluator:
    """Evaluate the real tax RAG chatbot with Phoenix tracing"""
    
    def __init__(self):
        """Initialize Phoenix tracer, LLM service, and evaluators"""
        print("🚀 Initializing real tax chatbot evaluation...")
        print("⏳ Loading models (this may take a moment)...")
        
        self.setup_phoenix_tracing()
        
        # Initialize the actual RAG system components
        try:
            # This will load the actual models
            llm_service.load_model()
            print("✅ LLM service loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: LLM service load issue: {e}")
        
        # Initialize evaluators
        self.tax_evaluator = TaxAccuracyEvaluator()
        self.hallucination_detector = HallucinationDetector()
        self.jurisdiction_evaluator = JurisdictionClassificationEvaluator()
        
        print("✅ Real tax chatbot evaluator initialized")
        print("🌍 Phoenix Dashboard: http://localhost:6006")
    
    def setup_phoenix_tracing(self):
        """Set up Phoenix tracing with OpenTelemetry"""
        try:
            # Configure OpenTelemetry
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()
            
            # Set up OTLP exporter to send to Phoenix
            otlp_exporter = OTLPSpanExporter(
                endpoint="http://localhost:6006/v1/traces",
                headers={}
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer("real-tax-rag-evaluation", "1.0.0")
            
            logger.info("Phoenix tracing configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Phoenix tracing: {e}")
            # Create a no-op tracer as fallback
            self.tracer = trace.NoOpTracer()
    
    def evaluate_real_query(self, query: str, expected_jurisdiction: str = "general") -> Dict[str, Any]:
        """Process a query through the real RAG system and evaluate it"""
        
        with self.tracer.start_as_current_span("real_tax_rag_evaluation") as main_span:
            evaluation_start = time.time()
            
            # Set main span attributes
            main_span.set_attributes({
                "service.name": "real-tax-rag-system",
                "evaluation.type": "end_to_end",
                "query.text": query[:200],
                "query.length": len(query),
                "expected_jurisdiction": expected_jurisdiction,
                "timestamp": datetime.now().isoformat()
            })
            
            try:
                print(f"🔍 Processing query: {query}")
                print("⏳ Running through RAG pipeline...")
                
                # ACTUAL RAG PROCESSING - This is the real tax chatbot
                rag_start_time = time.time()
                
                with self.tracer.start_as_current_span("rag_pipeline") as rag_span:
                    # Use the actual router-based query processing
                    result = llm_service.query_with_router(query)
                    
                    rag_duration = time.time() - rag_start_time
                    response = result.get('response', '')
                    query_type = result.get('query_type', 'unknown')
                    num_retrieved = result.get('num_retrieved', 0)
                    retrieval_strategy = result.get('retrieval_strategy', 'unknown')
                    user_context_found = result.get('user_context', {}).get('found', False) if 'user_context' in result else False
                    
                    # Add RAG pipeline metrics to span
                    rag_span.set_attributes({
                        "rag.duration_ms": rag_duration * 1000,
                        "rag.query_type": query_type,
                        "rag.documents_retrieved": num_retrieved,
                        "rag.retrieval_strategy": retrieval_strategy,
                        "rag.user_context_found": user_context_found,
                        "rag.response_length": len(response),
                        "rag.response_word_count": len(response.split()),
                        "rag.success": len(response) > 0
                    })
                    
                    rag_span.set_status(Status(StatusCode.OK))
                
                print(f"🤖 Generated response ({rag_duration:.2f}s): {response[:100]}...")
                print(f"📊 Query type: {query_type}, Retrieved docs: {num_retrieved}")
                
                if not response:
                    print("❌ Empty response from RAG system")
                    main_span.set_status(Status(StatusCode.ERROR, "Empty response"))
                    return {"error": "empty_response"}
                
                # EVALUATION PHASE
                evaluation_results = {}
                
                # 1. Tax Accuracy Evaluation
                print("📊 Evaluating tax accuracy...")
                with self.tracer.start_as_current_span("tax_accuracy_evaluation") as tax_span:
                    tax_eval_start = time.time()
                    
                    tax_result = self.tax_evaluator.evaluate_tax_response(
                        query=query,
                        response=response,
                        expected_jurisdiction=expected_jurisdiction
                    )
                    
                    tax_eval_duration = time.time() - tax_eval_start
                    
                    tax_span.set_attributes({
                        "evaluation.tax_accuracy": tax_result['overall_accuracy'],
                        "evaluation.tax_compliance": tax_result['tax_compliance'],
                        "evaluation.citation_accuracy": tax_result['citation_accuracy'],
                        "evaluation.response_completeness": tax_result['response_completeness'],
                        "evaluation.quality_issues_count": len(tax_result['quality_issues']),
                        "evaluation.duration_ms": tax_eval_duration * 1000
                    })
                    
                    # Add quality issues as events
                    for issue in tax_result['quality_issues']:
                        tax_span.add_event("quality_issue", {"issue.type": issue})
                    
                    tax_span.set_status(Status(StatusCode.OK))
                    evaluation_results['tax_accuracy'] = tax_result
                
                # 2. Hallucination Detection
                print("🚨 Checking for hallucinations...")
                with self.tracer.start_as_current_span("hallucination_detection") as hall_span:
                    hall_eval_start = time.time()
                    
                    hall_result = self.hallucination_detector.detect_hallucinations(
                        query=query,
                        response=response
                    )
                    
                    hall_eval_duration = time.time() - hall_eval_start
                    
                    hall_span.set_attributes({
                        "evaluation.hallucination_risk": hall_result['hallucination_risk'],
                        "evaluation.confidence_level": hall_result['confidence_level'],
                        "evaluation.recommended_action": hall_result['recommended_action'],
                        "evaluation.issues_count": len(hall_result['specific_issues']),
                        "evaluation.duration_ms": hall_eval_duration * 1000
                    })
                    
                    # Add specific issues as events
                    for issue in hall_result['specific_issues']:
                        hall_span.add_event("hallucination_issue", {"issue.description": issue})
                    
                    hall_span.set_status(Status(StatusCode.OK))
                    evaluation_results['hallucination'] = hall_result
                
                # 3. Jurisdiction Classification (if applicable)
                if expected_jurisdiction != "general" or any(keyword in query.lower() for keyword in ['california', 'ca', 'ftb']):
                    print("🗺️ Evaluating jurisdiction classification...")
                    with self.tracer.start_as_current_span("jurisdiction_evaluation") as juris_span:
                        juris_eval_start = time.time()
                        
                        # Determine predicted jurisdiction from RAG result
                        predicted_jurisdiction = "general"
                        if 'retrieval_strategy' in result and 'california' in result['retrieval_strategy'].lower():
                            predicted_jurisdiction = "california"
                        elif any(keyword in response.lower() for keyword in ['california', 'ca', 'ftb', 'franchise tax board']):
                            predicted_jurisdiction = "california"
                        
                        juris_accuracy = self.jurisdiction_evaluator.evaluate(
                            query=query,
                            response=response,
                            expected_jurisdiction=expected_jurisdiction
                        )
                        
                        is_correct = predicted_jurisdiction == expected_jurisdiction
                        juris_eval_duration = time.time() - juris_eval_start
                        
                        juris_span.set_attributes({
                            "evaluation.jurisdiction_accuracy": juris_accuracy,
                            "jurisdiction.predicted": predicted_jurisdiction,
                            "jurisdiction.expected": expected_jurisdiction,
                            "jurisdiction.is_correct": is_correct,
                            "evaluation.duration_ms": juris_eval_duration * 1000
                        })
                        
                        juris_span.add_event("jurisdiction_classified", {
                            "classification.predicted": predicted_jurisdiction,
                            "classification.expected": expected_jurisdiction,
                            "classification.correct": is_correct
                        })
                        
                        juris_span.set_status(Status(StatusCode.OK))
                        evaluation_results['jurisdiction'] = {
                            'accuracy': juris_accuracy,
                            'predicted': predicted_jurisdiction,
                            'expected': expected_jurisdiction,
                            'is_correct': is_correct
                        }
                
                # Calculate overall metrics
                total_duration = time.time() - evaluation_start
                overall_score = (
                    evaluation_results['tax_accuracy']['overall_accuracy'] * 0.6 +
                    (1 - evaluation_results['hallucination']['hallucination_risk']) * 0.2 +
                    evaluation_results.get('jurisdiction', {}).get('accuracy', 0.8) * 0.2
                )
                
                # Add summary to main span
                main_span.set_attributes({
                    "evaluation.overall_score": overall_score,
                    "evaluation.total_duration_ms": total_duration * 1000,
                    "evaluation.rag_duration_ms": rag_duration * 1000,
                    "evaluation.success": True
                })
                
                main_span.set_status(Status(StatusCode.OK))
                
                # Print results
                print(f"✅ Evaluation completed in {total_duration:.2f}s")
                print(f"📊 Tax Accuracy: {evaluation_results['tax_accuracy']['overall_accuracy']:.3f}")
                print(f"🚨 Hallucination Risk: {evaluation_results['hallucination']['hallucination_risk']:.3f}")
                if 'jurisdiction' in evaluation_results:
                    print(f"🗺️  Jurisdiction: {evaluation_results['jurisdiction']['predicted']} ({'✓' if evaluation_results['jurisdiction']['is_correct'] else '✗'})")
                print(f"🎯 Overall Score: {overall_score:.3f}")
                
                return {
                    'query': query,
                    'response': response,
                    'rag_duration': rag_duration,
                    'evaluation_results': evaluation_results,
                    'overall_score': overall_score,
                    'rag_metadata': {
                        'query_type': query_type,
                        'num_retrieved': num_retrieved,
                        'retrieval_strategy': retrieval_strategy,
                        'user_context_found': user_context_found
                    }
                }
                
            except Exception as e:
                error_msg = f"Evaluation failed: {str(e)}"
                print(f"❌ {error_msg}")
                
                main_span.set_attributes({
                    "evaluation.success": False,
                    "evaluation.error": str(e),
                    "evaluation.error_type": type(e).__name__
                })
                main_span.set_status(Status(StatusCode.ERROR, str(e)))
                
                import traceback
                traceback.print_exc()
                
                return {"error": error_msg, "exception": str(e)}
    
    def run_comprehensive_real_evaluation(self):
        """Run comprehensive evaluation on real tax queries"""
        
        print("🚀 REAL TAX CHATBOT EVALUATION")
        print("=" * 60)
        print("Processing queries through the actual RAG pipeline")
        print("🌍 View live traces at: http://localhost:6006")
        print("=" * 60)
        
        # Real tax queries to test
        test_queries = [
            {
                'query': 'What is business and occupation tax?',
                'expected_jurisdiction': 'general',
                'description': 'General B&O tax question'
            },
            {
                'query': 'What are California franchise tax requirements?',
                'expected_jurisdiction': 'california', 
                'description': 'California-specific tax question'
            },
            {
                'query': 'My name is Sarah Johnson and I need my tax information',
                'expected_jurisdiction': 'general',
                'description': 'Personal data query (should route to database)'
            },
            {
                'query': 'How do I calculate quarterly estimated taxes for my business?',
                'expected_jurisdiction': 'general',
                'description': 'Complex business tax calculation'
            },
            {
                'query': 'What is Publication 29 about in California?',
                'expected_jurisdiction': 'california',
                'description': 'California document reference'
            }
        ]
        
        results = []
        total_start_time = time.time()
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*20} Test {i}/{len(test_queries)} {'='*20}")
            print(f"📝 Description: {test_case['description']}")
            
            result = self.evaluate_real_query(
                query=test_case['query'],
                expected_jurisdiction=test_case['expected_jurisdiction']
            )
            
            results.append({
                'test_case': test_case,
                'result': result
            })
            
            # Wait between queries to see individual traces
            if i < len(test_queries):
                print("⏳ Waiting 5 seconds for trace processing...")
                time.sleep(5)
        
        total_time = time.time() - total_start_time
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        
        successful_evaluations = [r for r in results if 'error' not in r['result']]
        
        if successful_evaluations:
            avg_rag_time = sum(r['result']['rag_duration'] for r in successful_evaluations) / len(successful_evaluations)
            avg_overall_score = sum(r['result']['overall_score'] for r in successful_evaluations) / len(successful_evaluations)
            
            print(f"✅ Successful evaluations: {len(successful_evaluations)}/{len(results)}")
            print(f"⏱️  Average RAG time: {avg_rag_time:.2f}s")
            print(f"🎯 Average overall score: {avg_overall_score:.3f}")
            print(f"⏱️  Total evaluation time: {total_time:.2f}s")
            
            # Query type breakdown
            query_types = {}
            for r in successful_evaluations:
                qt = r['result']['rag_metadata']['query_type']
                query_types[qt] = query_types.get(qt, 0) + 1
            
            print(f"📊 Query type breakdown: {query_types}")
            
        else:
            print("❌ No successful evaluations")
        
        print(f"🌍 Phoenix Dashboard: http://localhost:6006")
        print("=" * 60)
        
        return results

def main():
    """Main execution function"""
    try:
        evaluator = RealTaxChatbotEvaluator()
        results = evaluator.run_comprehensive_real_evaluation()
        
        print("✅ Real tax chatbot evaluation completed!")
        print("🔍 Check Phoenix dashboard for detailed trace analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()