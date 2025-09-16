"""
Automated evaluation service for continuous monitoring of tax RAG system
Provides scheduled evaluation runs, performance tracking, and alerting
"""
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import schedule

from ..core.config import config_manager
from ..core.llm_service import llm_service
from ..core.phoenix_tracer import get_phoenix_tracer
from .tax_evaluators import (HallucinationDetector,
                             JurisdictionClassificationEvaluator,
                             TaxAccuracyEvaluator)

logger = logging.getLogger(__name__)


class AutomatedEvaluationService:
    """
    Service for running automated evaluations on the tax RAG system
    
    Features:
    - Scheduled daily/hourly evaluations
    - Real-time sampling of live queries
    - Performance tracking and trend analysis
    - Automated alerting for quality degradation
    - Integration with Phoenix monitoring
    """
    
    def __init__(self):
        self.config = config_manager.phoenix.evaluation if hasattr(config_manager, 'phoenix') and config_manager.phoenix else {}
        self.phoenix_tracer = get_phoenix_tracer()
        
        # Evaluation settings - handle both dict and object config
        if hasattr(self.config, 'enabled'):
            self.enabled = self.config.enabled
            self.auto_eval_interval = getattr(self.config, 'auto_eval_interval', 10)
            self.tax_accuracy_threshold = getattr(self.config, 'tax_accuracy_threshold', 0.85)
            self.hallucination_threshold = getattr(self.config, 'hallucination_threshold', 0.1)
            self.jurisdiction_threshold = getattr(self.config, 'jurisdiction_accuracy_threshold', 0.90)
        else:
            # Fallback for dict config or missing config
            self.enabled = True
            self.auto_eval_interval = 10
            self.tax_accuracy_threshold = 0.85
            self.hallucination_threshold = 0.1
            self.jurisdiction_threshold = 0.90
        
        # Initialize evaluators
        self.tax_accuracy_evaluator = TaxAccuracyEvaluator()
        self.hallucination_detector = HallucinationDetector()
        self.jurisdiction_evaluator = JurisdictionClassificationEvaluator()
        
        # Data storage
        self.evaluation_history = []
        self.live_query_buffer = []
        self.performance_trends = {}
        
        # Threading for background operations
        self.scheduler_thread = None
        self.is_running = False
        
        # Load evaluation datasets
        self.datasets = self._load_evaluation_datasets()
        
        logger.info(f"Automated evaluation service initialized (enabled: {self.enabled})")
    
    def _load_evaluation_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load evaluation datasets from JSON files"""
        datasets = {}
        dataset_dir = Path(__file__).parent / 'datasets'
        
        dataset_files = {
            'tax_accuracy': dataset_dir / 'tax_accuracy.json',
            'hallucination': dataset_dir / 'hallucination_prompts.json',
            'jurisdiction': dataset_dir / 'jurisdiction_tests.json'
        }
        
        for dataset_name, file_path in dataset_files.items():
            try:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        datasets[dataset_name] = json.load(f)
                    logger.info(f"Loaded {len(datasets[dataset_name])} test cases for {dataset_name}")
                else:
                    logger.warning(f"Dataset file not found: {file_path}")
                    datasets[dataset_name] = []
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
                datasets[dataset_name] = []
        
        return datasets
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run full evaluation suite across all test categories
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        if not self.enabled:
            logger.info("Automated evaluation is disabled")
            return {"status": "disabled"}
        
        logger.info(f"Starting comprehensive evaluation at {datetime.now()}")
        start_time = time.time()
        
        evaluation_report = {
            'evaluation_id': f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'comprehensive',
            'results': {},
            'summary': {},
            'alerts': [],
            'performance_metrics': {}
        }
        
        # Phoenix tracing for evaluation
        if self.phoenix_tracer and self.phoenix_tracer.enabled:
            with self.phoenix_tracer.tracer.start_as_current_span("automated_evaluation") as span:
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    operation_type="comprehensive_evaluation",
                    evaluation_id=evaluation_report['evaluation_id'],
                    num_test_datasets=len(self.datasets)
                ))
                
                evaluation_report = self._run_evaluation_internal(evaluation_report)
                
                # Add evaluation metrics to span
                span.set_attributes(self.phoenix_tracer._create_span_attributes(
                    overall_accuracy=evaluation_report['summary'].get('overall_accuracy', 0),
                    num_alerts=len(evaluation_report['alerts']),
                    evaluation_success=True
                ))
        else:
            evaluation_report = self._run_evaluation_internal(evaluation_report)
        
        # Store evaluation results
        self.evaluation_history.append(evaluation_report)
        self._save_evaluation_report(evaluation_report)
        
        # Update performance trends
        self._update_performance_trends(evaluation_report)
        
        # Check for alerts
        self._check_and_send_alerts(evaluation_report)
        
        duration = time.time() - start_time
        logger.info(f"Comprehensive evaluation completed in {duration:.2f}s - Overall accuracy: {evaluation_report['summary'].get('overall_accuracy', 0):.3f}")
        
        return evaluation_report
    
    def _run_evaluation_internal(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Internal evaluation execution"""
        
        # 1. Tax Accuracy Evaluation
        try:
            tax_accuracy_results = self._evaluate_tax_accuracy()
            evaluation_report['results']['tax_accuracy'] = tax_accuracy_results
            logger.info(f"Tax accuracy evaluation completed: {tax_accuracy_results['avg_accuracy']:.3f}")
        except Exception as e:
            logger.error(f"Tax accuracy evaluation failed: {str(e)}")
            evaluation_report['results']['tax_accuracy'] = {'error': str(e)}
        
        # 2. Hallucination Detection
        try:
            hallucination_results = self._evaluate_hallucination_detection()
            evaluation_report['results']['hallucination'] = hallucination_results
            logger.info(f"Hallucination detection completed: {hallucination_results['avg_risk']:.3f}")
        except Exception as e:
            logger.error(f"Hallucination evaluation failed: {str(e)}")
            evaluation_report['results']['hallucination'] = {'error': str(e)}
        
        # 3. Jurisdiction Classification
        try:
            jurisdiction_results = self._evaluate_jurisdiction_classification()
            evaluation_report['results']['jurisdiction'] = jurisdiction_results
            logger.info(f"Jurisdiction classification completed: {jurisdiction_results['accuracy']:.3f}")
        except Exception as e:
            logger.error(f"Jurisdiction evaluation failed: {str(e)}")
            evaluation_report['results']['jurisdiction'] = {'error': str(e)}
        
        # 4. Generate summary
        evaluation_report['summary'] = self._generate_evaluation_summary(evaluation_report['results'])
        
        return evaluation_report
    
    def _evaluate_tax_accuracy(self) -> Dict[str, Any]:
        """Evaluate tax accuracy using the test dataset"""
        test_cases = self.datasets['tax_accuracy'][:20]  # Limit for performance
        
        queries = []
        responses = []
        expected_jurisdictions = []
        
        for test_case in test_cases:
            query = test_case['query']
            expected_jurisdiction = test_case.get('expected_jurisdiction', 'general')
            
            # Generate response using the RAG system
            try:
                result = llm_service.query_with_router(query)
                response = result.get('response', '')
            except Exception as e:
                logger.warning(f"Failed to generate response for query: {query[:50]}... Error: {str(e)}")
                response = ''
            
            queries.append(query)
            responses.append(response)
            expected_jurisdictions.append(expected_jurisdiction)
        
        # Evaluate using tax accuracy evaluator
        evaluation_df = tax_accuracy_evaluator.evaluate_batch(
            queries=queries,
            responses=responses,
            expected_jurisdictions=expected_jurisdictions
        )
        
        # Calculate aggregate metrics
        avg_accuracy = evaluation_df['overall_accuracy'].mean()
        avg_jurisdiction_accuracy = evaluation_df['jurisdiction_accuracy'].mean()
        avg_citation_accuracy = evaluation_df['citation_accuracy'].mean()
        avg_compliance = evaluation_df['tax_compliance'].mean()
        
        # Quality issues analysis
        quality_issues = []
        for issues_str in evaluation_df['quality_issues']:
            if issues_str:
                quality_issues.extend(issues_str.split(', '))
        
        issue_counts = {}
        for issue in quality_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'avg_accuracy': avg_accuracy,
            'avg_jurisdiction_accuracy': avg_jurisdiction_accuracy,
            'avg_citation_accuracy': avg_citation_accuracy,
            'avg_compliance': avg_compliance,
            'num_test_cases': len(test_cases),
            'quality_issues': issue_counts,
            'detailed_results': evaluation_df.to_dict('records')
        }
    
    def _evaluate_hallucination_detection(self) -> Dict[str, Any]:
        """Evaluate hallucination detection using test prompts"""
        test_cases = self.datasets['hallucination'][:15]  # Limit for performance
        
        hallucination_results = []
        
        for test_case in test_cases:
            query = test_case['query']
            expected_risk = test_case.get('hallucination_risk', 'medium')
            
            # Generate response
            try:
                result = llm_service.query_with_router(query)
                response = result.get('response', '')
            except Exception as e:
                logger.warning(f"Failed to generate response for hallucination test: {query[:50]}...")
                response = 'Error generating response'
            
            # Detect hallucinations
            hallucination_analysis = hallucination_detector.detect_hallucinations(query, response)
            
            hallucination_results.append({
                'query': query,
                'response': response[:200] + '...' if len(response) > 200 else response,
                'expected_risk': expected_risk,
                'detected_risk': hallucination_analysis['hallucination_risk'],
                'confidence_level': hallucination_analysis['confidence_level'],
                'specific_issues': hallucination_analysis['specific_issues'],
                'recommended_action': hallucination_analysis['recommended_action']
            })
        
        # Calculate aggregate metrics
        avg_risk = sum(r['detected_risk'] for r in hallucination_results) / len(hallucination_results)
        high_risk_count = sum(1 for r in hallucination_results if r['detected_risk'] >= 0.7)
        
        return {
            'avg_risk': avg_risk,
            'high_risk_count': high_risk_count,
            'num_test_cases': len(test_cases),
            'detailed_results': hallucination_results
        }
    
    def _evaluate_jurisdiction_classification(self) -> Dict[str, Any]:
        """Evaluate jurisdiction classification accuracy"""
        test_cases = self.datasets['jurisdiction'][:25]  # Limit for performance
        
        queries = []
        true_jurisdictions = []
        predicted_jurisdictions = []
        
        for test_case in test_cases:
            query = test_case['query']
            expected_jurisdiction = test_case['expected_jurisdiction']
            
            queries.append(query)
            true_jurisdictions.append(expected_jurisdiction)
            
            # Get predicted jurisdiction through RAG system
            try:
                result = llm_service.query_with_router(query)
                # Extract jurisdiction from result metadata
                jurisdiction = 'general'  # default
                
                if 'retrieval_strategy' in result:
                    strategy = result['retrieval_strategy']
                    if 'california' in strategy:
                        jurisdiction = 'california'
                    elif 'general' in strategy:
                        jurisdiction = 'general'
                
                predicted_jurisdictions.append(jurisdiction)
                
            except Exception as e:
                logger.warning(f"Failed to classify jurisdiction for: {query[:50]}...")
                predicted_jurisdictions.append('general')  # default fallback
        
        # Evaluate classification accuracy
        classification_results = jurisdiction_evaluator.evaluate_jurisdiction_classification(
            queries=queries,
            predicted_jurisdictions=predicted_jurisdictions,
            true_jurisdictions=true_jurisdictions
        )
        
        return classification_results
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all evaluation results"""
        summary = {
            'overall_accuracy': 0.0,
            'key_metrics': {},
            'performance_status': 'unknown',
            'recommendations': []
        }
        
        # Aggregate key metrics
        if 'tax_accuracy' in results and 'avg_accuracy' in results['tax_accuracy']:
            summary['key_metrics']['tax_accuracy'] = results['tax_accuracy']['avg_accuracy']
        
        if 'hallucination' in results and 'avg_risk' in results['hallucination']:
            summary['key_metrics']['hallucination_risk'] = results['hallucination']['avg_risk']
        
        if 'jurisdiction' in results and 'overall_accuracy' in results['jurisdiction']:
            summary['key_metrics']['jurisdiction_accuracy'] = results['jurisdiction']['overall_accuracy']
        
        # Calculate weighted overall accuracy
        weights = {'tax_accuracy': 0.5, 'jurisdiction_accuracy': 0.3, 'hallucination_risk': 0.2}
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in summary['key_metrics']:
                if metric == 'hallucination_risk':
                    # Convert risk to accuracy (lower risk = higher accuracy)
                    score = 1.0 - summary['key_metrics'][metric]
                else:
                    score = summary['key_metrics'][metric]
                weighted_score += score * weight
                total_weight += weight
        
        summary['overall_accuracy'] = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine performance status
        if summary['overall_accuracy'] >= 0.9:
            summary['performance_status'] = 'excellent'
        elif summary['overall_accuracy'] >= 0.8:
            summary['performance_status'] = 'good'
        elif summary['overall_accuracy'] >= 0.7:
            summary['performance_status'] = 'acceptable'
        else:
            summary['performance_status'] = 'needs_improvement'
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(results, summary)
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any], summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Tax accuracy recommendations
        if 'tax_accuracy' in results:
            tax_accuracy = results['tax_accuracy'].get('avg_accuracy', 0)
            if tax_accuracy < self.tax_accuracy_threshold:
                recommendations.append(f"Tax accuracy ({tax_accuracy:.3f}) below threshold ({self.tax_accuracy_threshold}). Consider improving document retrieval or response generation.")
        
        # Hallucination recommendations
        if 'hallucination' in results:
            hallucination_risk = results['hallucination'].get('avg_risk', 0)
            if hallucination_risk > self.hallucination_threshold:
                recommendations.append(f"Hallucination risk ({hallucination_risk:.3f}) above threshold ({self.hallucination_threshold}). Review response generation prompts and add more disclaimers.")
        
        # Jurisdiction recommendations
        if 'jurisdiction' in results:
            jurisdiction_accuracy = results['jurisdiction'].get('overall_accuracy', 0)
            if jurisdiction_accuracy < self.jurisdiction_threshold:
                recommendations.append(f"Jurisdiction accuracy ({jurisdiction_accuracy:.3f}) below threshold ({self.jurisdiction_threshold}). Improve query classification model.")
        
        # Overall performance recommendations
        if summary['overall_accuracy'] < 0.8:
            recommendations.append("Overall system performance needs improvement. Consider comprehensive review of RAG pipeline components.")
        
        return recommendations
    
    def _save_evaluation_report(self, report: Dict[str, Any]) -> None:
        """Save evaluation report to file"""
        try:
            # Create reports directory
            reports_dir = Path("./data/evaluation_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed report
            report_file = reports_dir / f"{report['evaluation_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save summary to CSV for trend analysis
            summary_file = reports_dir / "evaluation_summary.csv"
            summary_row = {
                'timestamp': report['timestamp'],
                'evaluation_id': report['evaluation_id'],
                'overall_accuracy': report['summary'].get('overall_accuracy', 0),
                'tax_accuracy': report['summary'].get('key_metrics', {}).get('tax_accuracy', 0),
                'jurisdiction_accuracy': report['summary'].get('key_metrics', {}).get('jurisdiction_accuracy', 0),
                'hallucination_risk': report['summary'].get('key_metrics', {}).get('hallucination_risk', 0),
                'num_alerts': len(report['alerts'])
            }
            
            # Append to CSV
            df = pd.DataFrame([summary_row])
            if summary_file.exists():
                df.to_csv(summary_file, mode='a', header=False, index=False)
            else:
                df.to_csv(summary_file, index=False)
                
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {str(e)}")
    
    def _update_performance_trends(self, report: Dict[str, Any]) -> None:
        """Update performance trend tracking"""
        timestamp = datetime.fromisoformat(report['timestamp'].replace('Z', '+00:00'))
        
        metrics = report['summary'].get('key_metrics', {})
        for metric_name, value in metrics.items():
            if metric_name not in self.performance_trends:
                self.performance_trends[metric_name] = []
            
            self.performance_trends[metric_name].append({
                'timestamp': timestamp,
                'value': value
            })
            
            # Keep only recent data (last 30 days)
            cutoff = timestamp - timedelta(days=30)
            self.performance_trends[metric_name] = [
                point for point in self.performance_trends[metric_name]
                if point['timestamp'] > cutoff
            ]
    
    def _check_and_send_alerts(self, report: Dict[str, Any]) -> None:
        """Check evaluation results against thresholds and send alerts"""
        alerts = []
        
        # Check tax accuracy threshold
        tax_accuracy = report['summary'].get('key_metrics', {}).get('tax_accuracy', 1.0)
        if tax_accuracy < self.tax_accuracy_threshold:
            alerts.append({
                'type': 'accuracy_drop',
                'severity': 'high',
                'metric': 'tax_accuracy',
                'value': tax_accuracy,
                'threshold': self.tax_accuracy_threshold,
                'message': f"Tax accuracy dropped to {tax_accuracy:.3f} (threshold: {self.tax_accuracy_threshold})"
            })
        
        # Check hallucination risk threshold
        hallucination_risk = report['summary'].get('key_metrics', {}).get('hallucination_risk', 0.0)
        if hallucination_risk > self.hallucination_threshold:
            alerts.append({
                'type': 'high_hallucination',
                'severity': 'critical',
                'metric': 'hallucination_risk',
                'value': hallucination_risk,
                'threshold': self.hallucination_threshold,
                'message': f"Hallucination risk increased to {hallucination_risk:.3f} (threshold: {self.hallucination_threshold})"
            })
        
        # Check jurisdiction accuracy threshold
        jurisdiction_accuracy = report['summary'].get('key_metrics', {}).get('jurisdiction_accuracy', 1.0)
        if jurisdiction_accuracy < self.jurisdiction_threshold:
            alerts.append({
                'type': 'jurisdiction_misclassification',
                'severity': 'medium',
                'metric': 'jurisdiction_accuracy',
                'value': jurisdiction_accuracy,
                'threshold': self.jurisdiction_threshold,
                'message': f"Jurisdiction accuracy dropped to {jurisdiction_accuracy:.3f} (threshold: {self.jurisdiction_threshold})"
            })
        
        report['alerts'] = alerts
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")
            
            # Send to Phoenix monitoring if available
            if self.phoenix_tracer and self.phoenix_tracer.enabled:
                self.phoenix_tracer.log_tax_accuracy_metrics(
                    accuracy_score=tax_accuracy,
                    jurisdiction_accuracy=jurisdiction_accuracy,
                    citation_accuracy=0.0,  # Could be added from detailed results
                    alert_type=alert['type'],
                    alert_severity=alert['severity']
                )
    
    def _run_quick_accuracy_check(self) -> Dict[str, Any]:
        """Run quick accuracy check with subset of test cases"""
        logger.info("Running quick accuracy check...")
        
        # Use smaller subset for quick check
        quick_test_cases = self.datasets['tax_accuracy'][:5]
        
        if not quick_test_cases:
            return {"status": "no_test_cases", "accuracy": 0.0}
        
        total_score = 0.0
        for test_case in quick_test_cases:
            try:
                query = test_case['query']
                result = llm_service.query_with_router(query)
                response = result.get('response', '')
                
                # Quick evaluation
                evaluation = tax_accuracy_evaluator.evaluate_tax_response(
                    query=query,
                    response=response,
                    expected_jurisdiction=test_case.get('expected_jurisdiction')
                )
                
                total_score += evaluation['overall_accuracy']
                
            except Exception as e:
                logger.warning(f"Quick check failed for query: {str(e)}")
        
        avg_accuracy = total_score / len(quick_test_cases)
        
        logger.info(f"Quick accuracy check completed: {avg_accuracy:.3f}")
        return {"status": "completed", "accuracy": avg_accuracy, "num_cases": len(quick_test_cases)}
    
    def setup_scheduled_evaluations(self) -> None:
        """Set up automated evaluation schedule"""
        if not self.enabled:
            logger.info("Scheduled evaluations disabled")
            return
        
        schedule_config = self.config.get('schedule', {})
        
        # Daily comprehensive evaluation
        daily_time = schedule_config.get('daily_comprehensive', '02:00')
        schedule.every().day.at(daily_time).do(self.run_comprehensive_evaluation)
        logger.info(f"Scheduled daily comprehensive evaluation at {daily_time}")
        
        # Hourly quick accuracy check
        if schedule_config.get('hourly_quick_check') == 'every_hour':
            schedule.every().hour.do(self._run_quick_accuracy_check)
            logger.info("Scheduled hourly quick accuracy checks")
        
        logger.info("Automated evaluation scheduling configured")
    
    def start_evaluation_scheduler(self) -> None:
        """Start the evaluation scheduler in background thread"""
        if not self.enabled:
            logger.info("Evaluation scheduler disabled")
            return
        
        if self.is_running:
            logger.warning("Evaluation scheduler already running")
            return
        
        self.is_running = True
        
        def scheduler_loop():
            logger.info("Starting evaluation scheduler...")
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Scheduler error: {str(e)}")
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Evaluation scheduler started in background thread")
    
    def stop_evaluation_scheduler(self) -> None:
        """Stop the evaluation scheduler"""
        if self.is_running:
            self.is_running = False
            logger.info("Evaluation scheduler stopped")
    
    def get_recent_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for recent period"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_evaluations = [
            eval_report for eval_report in self.evaluation_history[-20:]  # Last 20 evaluations
            if datetime.fromisoformat(eval_report['timestamp']) > cutoff
        ]
        
        if not recent_evaluations:
            return {"status": "no_recent_data", "days": days}
        
        # Calculate averages
        avg_accuracy = sum(
            eval_report['summary'].get('overall_accuracy', 0)
            for eval_report in recent_evaluations
        ) / len(recent_evaluations)
        
        total_alerts = sum(len(eval_report['alerts']) for eval_report in recent_evaluations)
        
        return {
            "status": "success",
            "period_days": days,
            "num_evaluations": len(recent_evaluations),
            "avg_overall_accuracy": avg_accuracy,
            "total_alerts": total_alerts,
            "latest_evaluation": recent_evaluations[-1]['timestamp'] if recent_evaluations else None,
            "performance_trend": "improving" if len(recent_evaluations) >= 2 and 
                               recent_evaluations[-1]['summary'].get('overall_accuracy', 0) > 
                               recent_evaluations[-2]['summary'].get('overall_accuracy', 0) else "stable"
        }


# Global automated evaluation service instance
automated_evaluation_service = AutomatedEvaluationService()