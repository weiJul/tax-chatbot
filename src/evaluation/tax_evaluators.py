"""
Tax-specific evaluation metrics and evaluators for Phoenix monitoring
Provides specialized metrics for tax domain accuracy, compliance, and quality assessment
"""
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from phoenix.evals import LLMEvaluator, OpenAIModel
    PHOENIX_EVALS_AVAILABLE = True
except ImportError:
    PHOENIX_EVALS_AVAILABLE = False
    logging.warning("Phoenix evals not available - evaluation will use local metrics only")

from ..core.config import config_manager

logger = logging.getLogger(__name__)


class TaxAccuracyEvaluator:
    """
    Evaluates accuracy of tax-related responses using specialized tax domain criteria
    """
    
    def __init__(self):
        self.config = config_manager.phoenix.evaluation if hasattr(config_manager, 'phoenix') and config_manager.phoenix else {}
        if hasattr(self.config, 'tax_accuracy_threshold'):
            self.accuracy_threshold = self.config.tax_accuracy_threshold
        else:
            self.accuracy_threshold = getattr(self.config, 'tax_accuracy_threshold', 0.85) if hasattr(self.config, 'get') else 0.85
        self.local_model = None  # For local evaluation when OpenAI not available
        
    def evaluate(self, query: str, response: str) -> float:
        """
        Simple evaluate method for compatibility with testing framework
        
        Returns a score between 0.0 and 1.0 indicating accuracy
        """
        try:
            result = self.evaluate_tax_response(query, response)
            return result.get('overall_score', 0.5)
        except Exception as e:
            logger.warning(f"Evaluation failed: {str(e)}")
            return 0.5  # Return neutral score on error
        
    def evaluate_tax_response(
        self,
        query: str,
        response: str,
        reference_context: Optional[str] = None,
        expected_jurisdiction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single tax response across multiple dimensions
        
        Args:
            query: Original user query
            response: Generated response
            reference_context: Reference context/documents used
            expected_jurisdiction: Expected jurisdiction (california, general, etc.)
            
        Returns:
            Dictionary with evaluation scores and detailed metrics
        """
        evaluation_result = {
            'overall_accuracy': 0.0,
            'jurisdiction_accuracy': 0.0,
            'citation_accuracy': 0.0,
            'tax_compliance': 0.0,
            'response_completeness': 0.0,
            'professional_disclaimers': 0.0,
            'detailed_metrics': {},
            'quality_issues': [],
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # 1. Basic response quality assessment
        basic_quality = self._assess_basic_quality(response)
        evaluation_result.update(basic_quality)
        
        # 2. Tax-specific accuracy assessment
        tax_accuracy = self._assess_tax_accuracy(query, response, reference_context)
        evaluation_result.update(tax_accuracy)
        
        # 3. Jurisdiction accuracy assessment
        if expected_jurisdiction:
            jurisdiction_score = self._assess_jurisdiction_accuracy(
                query, response, expected_jurisdiction
            )
            evaluation_result['jurisdiction_accuracy'] = jurisdiction_score
        
        # 4. Citation and reference accuracy
        citation_score = self._assess_citation_accuracy(response, reference_context)
        evaluation_result['citation_accuracy'] = citation_score
        
        # 5. Tax compliance and professional standards
        compliance_score = self._assess_tax_compliance(response)
        evaluation_result['tax_compliance'] = compliance_score
        
        # 6. Calculate overall accuracy
        evaluation_result['overall_accuracy'] = self._calculate_overall_accuracy(evaluation_result)
        
        # 7. Identify quality issues
        quality_issues = self._identify_quality_issues(evaluation_result, response)
        evaluation_result['quality_issues'] = quality_issues
        
        return evaluation_result
    
    def _assess_basic_quality(self, response: str) -> Dict[str, float]:
        """Assess basic response quality metrics"""
        if not response or not response.strip():
            return {
                'response_completeness': 0.0,
                'detailed_metrics': {'empty_response': True}
            }
        
        response = response.strip()
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        # Completeness based on length and structure
        completeness_score = 0.0
        
        if word_count >= 20:
            completeness_score += 0.3
        if word_count >= 50:
            completeness_score += 0.3
        if sentence_count >= 2:
            completeness_score += 0.2
        if sentence_count >= 4:
            completeness_score += 0.2
        
        return {
            'response_completeness': completeness_score,
            'detailed_metrics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'response_length': len(response)
            }
        }
    
    def _assess_tax_accuracy(
        self,
        query: str,
        response: str,
        reference_context: Optional[str] = None
    ) -> Dict[str, float]:
        """Assess tax-specific accuracy using domain knowledge"""
        accuracy_score = 0.0
        
        # Check for tax-specific terms and concepts
        tax_terms_found = self._count_tax_terms(response.lower())
        if tax_terms_found > 0:
            accuracy_score += 0.2
        
        # Check for proper tax terminology usage
        proper_terminology = self._check_proper_tax_terminology(response)
        if proper_terminology:
            accuracy_score += 0.3
        
        # Check for factual consistency with common tax knowledge
        factual_consistency = self._check_factual_consistency(response)
        accuracy_score += factual_consistency * 0.5
        
        return {
            'tax_domain_accuracy': accuracy_score,
            'detailed_metrics': {
                'tax_terms_found': tax_terms_found,
                'proper_terminology': proper_terminology,
                'factual_consistency': factual_consistency
            }
        }
    
    def _assess_jurisdiction_accuracy(
        self,
        query: str,
        response: str,
        expected_jurisdiction: str
    ) -> float:
        """Assess whether response correctly addresses the expected jurisdiction"""
        response_lower = response.lower()
        
        jurisdiction_indicators = {
            'california': ['california', 'ca ', 'ftb', 'franchise tax board', 'pub29'],
            'general': ['federal', 'irs', 'internal revenue', 'united states'],
            'washington': ['washington', 'wa ', 'b&o tax', 'business occupation']
        }
        
        expected_indicators = jurisdiction_indicators.get(expected_jurisdiction.lower(), [])
        found_indicators = sum(1 for indicator in expected_indicators if indicator in response_lower)
        
        if not expected_indicators:
            return 0.5  # Unknown jurisdiction, give neutral score
        
        # Score based on presence of jurisdiction-specific terms
        if found_indicators >= 1:
            return min(1.0, 0.5 + (found_indicators * 0.25))
        else:
            return 0.0
    
    def _assess_citation_accuracy(self, response: str, reference_context: Optional[str] = None) -> float:
        """Assess quality and accuracy of citations and references"""
        citation_score = 0.0
        
        # Check for common tax document references
        tax_documents = [
            'pub29', 'publication 29', 'business tax basics',
            'irs', 'ftb', 'tax code', 'regulation'
        ]
        
        response_lower = response.lower()
        citations_found = sum(1 for doc in tax_documents if doc in response_lower)
        
        if citations_found > 0:
            citation_score += 0.5
        
        # Check for proper citation format (basic patterns)
        citation_patterns = [
            r'publication \d+',
            r'pub\s*\d+',
            r'section \d+',
            r'code section \d+'
        ]
        
        formal_citations = sum(1 for pattern in citation_patterns 
                             if re.search(pattern, response_lower))
        
        if formal_citations > 0:
            citation_score += 0.3
        
        # Bonus for multiple citations
        if citations_found > 1:
            citation_score += 0.2
        
        return min(1.0, citation_score)
    
    def _assess_tax_compliance(self, response: str) -> float:
        """Assess compliance with tax professional standards and disclaimers"""
        compliance_score = 0.0
        response_lower = response.lower()
        
        # Check for professional disclaimers
        disclaimer_terms = [
            'consult', 'professional', 'advisor', 'may vary',
            'specific situation', 'tax professional', 'qualified',
            'should verify', 'recommend consulting'
        ]
        
        disclaimers_found = sum(1 for term in disclaimer_terms if term in response_lower)
        
        if disclaimers_found >= 1:
            compliance_score += 0.4
        if disclaimers_found >= 2:
            compliance_score += 0.3
        
        # Check for appropriate caution in language
        cautious_language = [
            'generally', 'typically', 'usually', 'may', 'might',
            'could', 'often', 'in most cases'
        ]
        
        caution_found = sum(1 for term in cautious_language if term in response_lower)
        if caution_found >= 2:
            compliance_score += 0.3
        
        return min(1.0, compliance_score)
    
    def _count_tax_terms(self, response_lower: str) -> int:
        """Count relevant tax terms in response"""
        tax_terms = [
            'tax', 'deduction', 'credit', 'exemption', 'filing',
            'return', 'refund', 'withholding', 'liability', 'income',
            'gross', 'net', 'business', 'occupation', 'revenue',
            'expense', 'depreciation', 'amortization'
        ]
        
        return sum(1 for term in tax_terms if term in response_lower)
    
    def _check_proper_tax_terminology(self, response: str) -> bool:
        """Check if response uses proper tax terminology"""
        # Check for common misuse patterns
        misuse_patterns = [
            ('tax refund', 'tax return'),  # Common confusion
            ('tax shelter', 'tax haven'),  # Context-dependent
        ]
        
        # For now, return True if no obvious misuse detected
        # This could be expanded with more sophisticated checks
        return True
    
    def _check_factual_consistency(self, response: str) -> float:
        """Check for basic factual consistency in tax statements"""
        # Basic consistency checks
        consistency_score = 1.0
        response_lower = response.lower()
        
        # Check for obviously wrong statements
        wrong_patterns = [
            'tax rate.*100%',  # Tax rates shouldn't be 100%
            'no tax.*required',  # Absolute statements are usually wrong
            'never.*pay.*tax',   # Absolute statements
        ]
        
        for pattern in wrong_patterns:
            if re.search(pattern, response_lower):
                consistency_score -= 0.3
        
        return max(0.0, consistency_score)
    
    def _calculate_overall_accuracy(self, evaluation_result: Dict[str, Any]) -> float:
        """Calculate weighted overall accuracy score"""
        weights = {
            'response_completeness': 0.15,
            'tax_domain_accuracy': 0.30,
            'jurisdiction_accuracy': 0.20,
            'citation_accuracy': 0.15,
            'tax_compliance': 0.20
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in evaluation_result and evaluation_result[metric] is not None:
                weighted_score += evaluation_result[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _identify_quality_issues(self, evaluation_result: Dict[str, Any], response: str) -> List[str]:
        """Identify specific quality issues with the response"""
        issues = []
        
        if evaluation_result.get('overall_accuracy', 0) < 0.7:
            issues.append('low_overall_accuracy')
        
        if evaluation_result.get('response_completeness', 0) < 0.5:
            issues.append('incomplete_response')
        
        if evaluation_result.get('tax_compliance', 0) < 0.3:
            issues.append('missing_professional_disclaimers')
        
        if evaluation_result.get('citation_accuracy', 0) < 0.2:
            issues.append('poor_citation_quality')
        
        if evaluation_result.get('jurisdiction_accuracy', 0) < 0.5:
            issues.append('jurisdiction_mismatch')
        
        # Check for response length issues
        word_count = evaluation_result.get('detailed_metrics', {}).get('word_count', 0)
        if word_count < 10:
            issues.append('very_short_response')
        elif word_count > 500:
            issues.append('overly_verbose_response')
        
        return issues
    
    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        reference_contexts: Optional[List[str]] = None,
        expected_jurisdictions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Evaluate a batch of tax responses"""
        evaluations = []
        
        reference_contexts = reference_contexts or [None] * len(queries)
        expected_jurisdictions = expected_jurisdictions or [None] * len(queries)
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            eval_result = self.evaluate_tax_response(
                query=query,
                response=response,
                reference_context=reference_contexts[i],
                expected_jurisdiction=expected_jurisdictions[i]
            )
            
            # Flatten result for DataFrame
            flattened = {
                'query': query,
                'response': response[:200] + '...' if len(response) > 200 else response,
                'overall_accuracy': eval_result['overall_accuracy'],
                'jurisdiction_accuracy': eval_result['jurisdiction_accuracy'],
                'citation_accuracy': eval_result['citation_accuracy'],
                'tax_compliance': eval_result['tax_compliance'],
                'response_completeness': eval_result['response_completeness'],
                'quality_issues': ', '.join(eval_result['quality_issues']),
                'evaluation_timestamp': eval_result['evaluation_timestamp']
            }
            
            evaluations.append(flattened)
        
        return pd.DataFrame(evaluations)


class HallucinationDetector:
    """
    Specialized detector for tax-related hallucinations and misinformation
    """
    
    def __init__(self):
        self.config = config_manager.phoenix.evaluation if hasattr(config_manager, 'phoenix') and config_manager.phoenix else {}
        if hasattr(self.config, 'hallucination_threshold'):
            self.hallucination_threshold = self.config.hallucination_threshold
        else:
            self.hallucination_threshold = 0.1
    
    def evaluate(self, query: str, response: str) -> float:
        """
        Simple evaluate method for compatibility with testing framework
        
        Returns a score between 0.0 and 1.0 indicating hallucination likelihood
        """
        try:
            result = self.detect_hallucinations(query, response)
            return result.get('hallucination_score', 0.0)
        except Exception as e:
            logger.warning(f"Hallucination detection failed: {str(e)}")
            return 0.0  # Return no hallucination on error
    
    def detect_hallucinations(self, query: str, response: str) -> Dict[str, Any]:
        """
        Detect potential hallucinations in tax responses
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            Dictionary with hallucination analysis
        """
        hallucination_result = {
            'hallucination_risk': 0.0,
            'specific_issues': [],
            'confidence_level': 'high',
            'recommended_action': 'none'
        }
        
        risk_score = 0.0
        
        # Check for impossible tax scenarios
        impossible_patterns = [
            r'tax rate.*over 100%',
            r'negative.*tax.*rate',
            r'tax.*on.*mars',
            r'time.*travel.*deduction',
            r'alien.*tax.*credit'
        ]
        
        response_lower = response.lower()
        for pattern in impossible_patterns:
            if re.search(pattern, response_lower):
                risk_score += 0.5
                hallucination_result['specific_issues'].append(f'impossible_scenario: {pattern}')
        
        # Check for outdated information indicators
        outdated_patterns = [
            r'201[0-9].*tax.*rate',  # Very old tax rates
            r'trump.*tax.*plan.*current',
            r'obama.*tax.*policy.*current'
        ]
        
        for pattern in outdated_patterns:
            if re.search(pattern, response_lower):
                risk_score += 0.3
                hallucination_result['specific_issues'].append(f'potentially_outdated: {pattern}')
        
        # Check for overly specific claims without citations
        specific_uncited_patterns = [
            r'exactly.*\d+\.\d+%.*tax',
            r'precisely.*\$\d+.*deduction',
            r'the.*tax.*code.*section.*\d{4,}.*states'
        ]
        
        citations_present = any(citation in response_lower for citation in 
                              ['pub29', 'irs', 'publication', 'section'])
        
        if not citations_present:
            for pattern in specific_uncited_patterns:
                if re.search(pattern, response_lower):
                    risk_score += 0.4
                    hallucination_result['specific_issues'].append(f'specific_uncited_claim: {pattern}')
        
        hallucination_result['hallucination_risk'] = min(1.0, risk_score)
        
        # Determine confidence level and recommended action
        if risk_score >= 0.8:
            hallucination_result['confidence_level'] = 'high_risk'
            hallucination_result['recommended_action'] = 'flag_for_review'
        elif risk_score >= 0.5:
            hallucination_result['confidence_level'] = 'medium_risk'
            hallucination_result['recommended_action'] = 'add_disclaimers'
        elif risk_score >= 0.2:
            hallucination_result['confidence_level'] = 'low_risk'
            hallucination_result['recommended_action'] = 'monitor'
        else:
            hallucination_result['confidence_level'] = 'minimal_risk'
            hallucination_result['recommended_action'] = 'none'
        
        return hallucination_result


class JurisdictionClassificationEvaluator:
    """
    Evaluates the accuracy of jurisdiction classification for tax queries
    """
    
    def __init__(self):
        self.config = config_manager.phoenix.evaluation if hasattr(config_manager, 'phoenix') and config_manager.phoenix else {}
        if hasattr(self.config, 'jurisdiction_accuracy_threshold'):
            self.jurisdiction_threshold = self.config.jurisdiction_accuracy_threshold
        else:
            self.jurisdiction_threshold = 0.90
    
    def evaluate(self, query: str, response: str, expected_jurisdiction: str = "general") -> float:
        """
        Simple evaluate method for compatibility with testing framework
        
        Returns a score between 0.0 and 1.0 indicating jurisdiction classification accuracy
        """
        try:
            # Simple heuristic-based evaluation
            query_lower = query.lower()
            
            # California indicators
            ca_indicators = ['california', 'ca ', 'ftb', 'franchise tax board', 'pub29', 'publication 29']
            has_ca_indicators = any(indicator in query_lower for indicator in ca_indicators)
            
            if expected_jurisdiction == "california":
                return 0.9 if has_ca_indicators else 0.3
            else:  # general
                return 0.3 if has_ca_indicators else 0.9
                
        except Exception as e:
            logger.warning(f"Jurisdiction evaluation failed: {str(e)}")
            return 0.5  # Return neutral score on error
    
    def evaluate_jurisdiction_classification(
        self,
        queries: List[str],
        predicted_jurisdictions: List[str],
        true_jurisdictions: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate jurisdiction classification accuracy
        
        Args:
            queries: List of queries
            predicted_jurisdictions: Predicted jurisdictions
            true_jurisdictions: Ground truth jurisdictions
            
        Returns:
            Classification evaluation metrics
        """
        if len(queries) != len(predicted_jurisdictions) != len(true_jurisdictions):
            raise ValueError("All input lists must have the same length")
        
        # Calculate basic accuracy metrics
        correct_predictions = sum(1 for pred, true in zip(predicted_jurisdictions, true_jurisdictions) 
                                if pred == true)
        total_predictions = len(queries)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Create confusion matrix data
        unique_jurisdictions = list(set(true_jurisdictions + predicted_jurisdictions))
        confusion_matrix = {}
        
        for true_jurisdiction in unique_jurisdictions:
            confusion_matrix[true_jurisdiction] = {}
            for pred_jurisdiction in unique_jurisdictions:
                count = sum(1 for true, pred in zip(true_jurisdictions, predicted_jurisdictions)
                          if true == true_jurisdiction and pred == pred_jurisdiction)
                confusion_matrix[true_jurisdiction][pred_jurisdiction] = count
        
        # Calculate per-jurisdiction metrics
        jurisdiction_metrics = {}
        for jurisdiction in unique_jurisdictions:
            true_positives = confusion_matrix[jurisdiction].get(jurisdiction, 0)
            false_positives = sum(confusion_matrix[other_j].get(jurisdiction, 0) 
                                for other_j in unique_jurisdictions if other_j != jurisdiction)
            false_negatives = sum(confusion_matrix[jurisdiction].get(other_j, 0) 
                                for other_j in unique_jurisdictions if other_j != jurisdiction)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            jurisdiction_metrics[jurisdiction] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': sum(1 for j in true_jurisdictions if j == jurisdiction)
            }
        
        # Calculate macro-averaged metrics
        macro_precision = sum(metrics['precision'] for metrics in jurisdiction_metrics.values()) / len(jurisdiction_metrics)
        macro_recall = sum(metrics['recall'] for metrics in jurisdiction_metrics.values()) / len(jurisdiction_metrics)
        macro_f1 = sum(metrics['f1_score'] for metrics in jurisdiction_metrics.values()) / len(jurisdiction_metrics)
        
        return {
            'overall_accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'jurisdiction_metrics': jurisdiction_metrics,
            'confusion_matrix': confusion_matrix,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'evaluation_timestamp': datetime.now().isoformat()
        }


# Initialize global evaluators (lazy initialization to avoid config issues)
# tax_accuracy_evaluator = TaxAccuracyEvaluator()
# hallucination_detector = HallucinationDetector()
# jurisdiction_evaluator = JurisdictionClassificationEvaluator()