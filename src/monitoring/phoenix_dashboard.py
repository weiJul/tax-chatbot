"""
Phoenix dashboard configuration and management for tax RAG system
Provides custom dashboards, metrics visualization, and real-time monitoring
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import phoenix as px
    from phoenix.evals import run_evals
    from phoenix.trace import trace_dataset
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logging.warning("Phoenix not available - dashboard features will be limited")

from ..core.config import config_manager
from ..core.phoenix_tracer import get_phoenix_tracer

logger = logging.getLogger(__name__)


class TaxRagDashboard:
    """
    Custom Phoenix dashboard configuration and management for tax RAG system
    
    Features:
    - Tax-specific metric dashboards
    - Real-time performance monitoring
    - Custom alert rules and notifications
    - Performance trend visualization
    """
    
    def __init__(self):
        self.config = config_manager.phoenix.dashboard if hasattr(config_manager, 'phoenix') and config_manager.phoenix and hasattr(config_manager.phoenix, 'dashboard') else {}
        self.phoenix_tracer = get_phoenix_tracer()
        
        # Dashboard configuration - handle both dict and object config
        if hasattr(self.config, 'refresh_interval'):
            self.refresh_interval = self.config.refresh_interval
            self.retention_days = getattr(self.config, 'retention_days', 30)
            self.enable_alerts = getattr(self.config, 'enable_real_time_alerts', True)
            self.custom_views = getattr(self.config, 'custom_views', [])
        else:
            # Fallback for dict config or missing config
            self.refresh_interval = 5
            self.retention_days = 30
            self.enable_alerts = True
            self.custom_views = []
        
        # Alert rules
        self.alert_rules = self._load_alert_rules()
        
        logger.info("Tax RAG dashboard configuration initialized")
    
    def _load_alert_rules(self) -> Dict[str, Any]:
        """Load alert rules from configuration"""
        alert_config = config_manager.phoenix.alerts if hasattr(config_manager, 'phoenix') else {}
        return alert_config.get('rules', {})
    
    def setup_dashboard(self) -> Dict[str, Any]:
        """
        Configure Phoenix dashboard with tax-specific views and metrics
        
        Returns:
            Dashboard configuration dictionary
        """
        if not PHOENIX_AVAILABLE:
            logger.warning("Phoenix not available - cannot set up dashboard")
            return {"status": "phoenix_not_available"}
        
        logger.info("Setting up Phoenix dashboard for tax RAG system...")
        
        dashboard_config = {
            "dashboard_name": "Tax RAG System Monitoring",
            "refresh_interval_seconds": self.refresh_interval,
            "data_retention_days": self.retention_days,
            "views": self._create_dashboard_views(),
            "alerts": self._create_alert_configurations(),
            "metrics": self._define_custom_metrics(),
            "filters": self._create_dashboard_filters()
        }
        
        # Apply dashboard configuration
        try:
            self._apply_dashboard_configuration(dashboard_config)
            logger.info("Phoenix dashboard configuration applied successfully")
            return {"status": "success", "config": dashboard_config}
        except Exception as e:
            logger.error(f"Failed to apply dashboard configuration: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _create_dashboard_views(self) -> List[Dict[str, Any]]:
        """Create custom dashboard views for tax RAG monitoring"""
        views = []
        
        # 1. RAG Pipeline Performance View
        views.append({
            "name": "RAG Pipeline Performance",
            "description": "End-to-end RAG pipeline metrics and latency analysis",
            "type": "performance_dashboard",
            "metrics": [
                "retrieval_duration_ms",
                "generation_duration_ms", 
                "pipeline_success_rate",
                "end_to_end_latency_ms"
            ],
            "charts": [
                {
                    "type": "line_chart",
                    "title": "Pipeline Latency Over Time",
                    "metrics": ["retrieval_duration_ms", "generation_duration_ms"],
                    "time_range": "24h"
                },
                {
                    "type": "histogram",
                    "title": "Response Time Distribution",
                    "metric": "end_to_end_latency_ms",
                    "bins": 20
                },
                {
                    "type": "gauge",
                    "title": "Pipeline Success Rate",
                    "metric": "pipeline_success_rate",
                    "thresholds": {"good": 0.95, "warning": 0.90, "critical": 0.85}
                }
            ],
            "filters": ["time_range", "query_type", "jurisdiction"]
        })
        
        # 2. Tax Accuracy Metrics View
        views.append({
            "name": "Tax Accuracy Metrics",
            "description": "Tax domain-specific accuracy and quality metrics",
            "type": "accuracy_dashboard",
            "metrics": [
                "tax_accuracy_score",
                "jurisdiction_accuracy",
                "citation_accuracy",
                "tax_compliance_score"
            ],
            "charts": [
                {
                    "type": "gauge_panel",
                    "title": "Accuracy Scores",
                    "metrics": ["tax_accuracy_score", "jurisdiction_accuracy", "citation_accuracy"],
                    "thresholds": {"good": 0.85, "warning": 0.75, "critical": 0.65}
                },
                {
                    "type": "trend_chart",
                    "title": "Accuracy Trends (7 days)",
                    "metric": "tax_accuracy_score",
                    "time_range": "7d",
                    "aggregation": "daily_average"
                },
                {
                    "type": "bar_chart",
                    "title": "Accuracy by Jurisdiction",
                    "metric": "jurisdiction_accuracy",
                    "group_by": "jurisdiction"
                },
                {
                    "type": "heatmap",
                    "title": "Quality Issues Heatmap",
                    "metric": "quality_issues",
                    "dimensions": ["issue_type", "time_of_day"]
                }
            ],
            "filters": ["jurisdiction", "query_category", "time_range"]
        })
        
        # 3. User Experience View
        views.append({
            "name": "User Experience",
            "description": "User-facing metrics and query classification performance",
            "type": "user_experience_dashboard",
            "metrics": [
                "query_classification_accuracy",
                "user_lookup_success_rate",
                "context_enhancement_rate",
                "user_satisfaction_score"
            ],
            "charts": [
                {
                    "type": "confusion_matrix",
                    "title": "Query Classification Matrix",
                    "metric": "query_classification",
                    "actual_vs_predicted": True
                },
                {
                    "type": "funnel_chart",
                    "title": "User Context Pipeline",
                    "stages": ["query_received", "user_identified", "context_enhanced", "response_generated"]
                },
                {
                    "type": "line_chart",
                    "title": "User Lookup Success Rate",
                    "metric": "user_lookup_success_rate",
                    "time_range": "24h"
                }
            ],
            "filters": ["query_type", "user_found", "time_range"]
        })
        
        # 4. System Health View
        views.append({
            "name": "System Health",
            "description": "Infrastructure metrics and resource utilization",
            "type": "system_health_dashboard",
            "metrics": [
                "memory_gpu_usage_mb",
                "embedding_batch_size",
                "retrieval_cache_hit_rate",
                "model_load_time_ms"
            ],
            "charts": [
                {
                    "type": "area_chart",
                    "title": "GPU Memory Usage (RTX 2080 Ti)",
                    "metric": "memory_gpu_usage_mb",
                    "time_range": "1h",
                    "threshold_line": 9000  # 9GB warning threshold
                },
                {
                    "type": "stat_panel",
                    "title": "System Statistics",
                    "metrics": ["embedding_batch_size", "retrieval_cache_hit_rate"],
                    "display_mode": "current_value"
                },
                {
                    "type": "alert_panel",
                    "title": "Active Alerts",
                    "alert_types": ["memory_warning", "performance_degradation", "accuracy_drop"]
                }
            ],
            "filters": ["alert_severity", "time_range"]
        })
        
        # 5. Evaluation Results View
        views.append({
            "name": "Evaluation Results",
            "description": "Automated evaluation results and quality assessments",
            "type": "evaluation_dashboard",
            "metrics": [
                "overall_evaluation_score",
                "hallucination_risk",
                "evaluation_test_cases",
                "alert_frequency"
            ],
            "charts": [
                {
                    "type": "scorecard",
                    "title": "Latest Evaluation Results",
                    "metrics": ["overall_evaluation_score", "tax_accuracy_score", "hallucination_risk"]
                },
                {
                    "type": "timeline",
                    "title": "Evaluation History",
                    "metric": "evaluation_runs",
                    "time_range": "30d"
                },
                {
                    "type": "table",
                    "title": "Recent Quality Issues",
                    "columns": ["timestamp", "issue_type", "severity", "description", "resolution_status"]
                }
            ],
            "filters": ["evaluation_type", "severity", "time_range"]
        })
        
        return views
    
    def _create_alert_configurations(self) -> List[Dict[str, Any]]:
        """Create alert configurations for real-time monitoring"""
        alerts = []
        
        for rule_name, rule_config in self.alert_rules.items():
            alert_config = {
                "name": rule_name,
                "metric": rule_config.get('metric'),
                "threshold": rule_config.get('threshold'),
                "comparison": rule_config.get('comparison', 'above'),
                "severity": rule_config.get('severity', 'medium'),
                "description": self._get_alert_description(rule_name, rule_config),
                "notification_channels": self._get_notification_channels(),
                "evaluation_interval": "1m",
                "conditions": {
                    "duration": "5m",  # Alert after 5 minutes
                    "frequency": "once_per_hour"  # Don't spam
                }
            }
            alerts.append(alert_config)
        
        return alerts
    
    def _get_alert_description(self, rule_name: str, rule_config: Dict[str, Any]) -> str:
        """Generate alert description based on rule configuration"""
        metric = rule_config.get('metric', 'unknown')
        threshold = rule_config.get('threshold', 'N/A')
        comparison = rule_config.get('comparison', 'above')
        
        descriptions = {
            'accuracy_drop': f"Tax accuracy has dropped {comparison} {threshold}",
            'high_hallucination': f"Hallucination risk is {comparison} {threshold}",
            'jurisdiction_misclassification': f"Jurisdiction classification accuracy is {comparison} {threshold}",
            'gpu_memory_warning': f"GPU memory usage is {comparison} {threshold}MB",
            'response_latency': f"Response latency is {comparison} {threshold}ms"
        }
        
        return descriptions.get(rule_name, f"Metric {metric} is {comparison} threshold {threshold}")
    
    def _get_notification_channels(self) -> List[str]:
        """Get configured notification channels"""
        alert_config = config_manager.phoenix.alerts if hasattr(config_manager, 'phoenix') else {}
        channels = alert_config.get('channels', {})
        
        active_channels = []
        if channels.get('console', False):
            active_channels.append('console')
        if channels.get('log_file', False):
            active_channels.append('log_file')
        if channels.get('email', False):
            active_channels.append('email')
        if channels.get('slack', False):
            active_channels.append('slack')
        
        return active_channels or ['console']  # Default to console
    
    def _define_custom_metrics(self) -> List[Dict[str, Any]]:
        """Define custom metrics for tax RAG system"""
        metrics = [
            {
                "name": "tax_accuracy_score",
                "description": "Overall tax domain accuracy score",
                "type": "gauge",
                "unit": "percentage",
                "tags": ["accuracy", "tax", "quality"]
            },
            {
                "name": "jurisdiction_accuracy",
                "description": "Jurisdiction classification accuracy",
                "type": "gauge", 
                "unit": "percentage",
                "tags": ["classification", "jurisdiction", "accuracy"]
            },
            {
                "name": "hallucination_risk",
                "description": "Risk score for hallucinated responses",
                "type": "gauge",
                "unit": "risk_score",
                "tags": ["quality", "safety", "hallucination"]
            },
            {
                "name": "citation_accuracy",
                "description": "Accuracy of citations and references",
                "type": "gauge",
                "unit": "percentage",
                "tags": ["citations", "references", "accuracy"]
            },
            {
                "name": "tax_compliance_score",
                "description": "Compliance with tax professional standards",
                "type": "gauge",
                "unit": "percentage",
                "tags": ["compliance", "professional", "standards"]
            },
            {
                "name": "pipeline_success_rate",
                "description": "Rate of successful RAG pipeline executions",
                "type": "counter",
                "unit": "percentage",
                "tags": ["pipeline", "success", "reliability"]
            },
            {
                "name": "user_context_enhancement_rate",
                "description": "Rate of queries enhanced with user context",
                "type": "counter",
                "unit": "percentage",
                "tags": ["user", "context", "personalization"]
            }
        ]
        
        return metrics
    
    def _create_dashboard_filters(self) -> List[Dict[str, Any]]:
        """Create dashboard filters for data segmentation"""
        filters = [
            {
                "name": "time_range",
                "type": "time_picker",
                "options": ["5m", "15m", "1h", "6h", "24h", "7d", "30d"],
                "default": "1h"
            },
            {
                "name": "jurisdiction",
                "type": "multi_select",
                "options": ["california", "general", "washington"],
                "default": ["california", "general"]
            },
            {
                "name": "query_type",
                "type": "multi_select",
                "options": ["general_tax", "user_data", "general_search_with_context"],
                "default": ["general_tax", "user_data"]
            },
            {
                "name": "severity",
                "type": "single_select",
                "options": ["low", "medium", "high", "critical"],
                "default": "all"
            },
            {
                "name": "alert_type",
                "type": "multi_select",
                "options": ["accuracy_drop", "high_hallucination", "jurisdiction_misclassification", "gpu_memory_warning"],
                "default": "all"
            }
        ]
        
        return filters
    
    def _apply_dashboard_configuration(self, config: Dict[str, Any]) -> None:
        """Apply dashboard configuration to Phoenix"""
        # This would integrate with actual Phoenix API when available
        # For now, we log the configuration
        logger.info(f"Applying dashboard configuration: {config['dashboard_name']}")
        logger.info(f"Created {len(config['views'])} dashboard views")
        logger.info(f"Configured {len(config['alerts'])} alert rules")
        logger.info(f"Defined {len(config['metrics'])} custom metrics")
        
        # In a real implementation, this would use Phoenix API calls
        # px.configure_dashboard(config)
    
    def create_real_time_alerts(self) -> Dict[str, Any]:
        """Set up real-time alerting rules"""
        if not self.enable_alerts:
            return {"status": "alerts_disabled"}
        
        alert_configurations = []
        
        for rule_name, rule_config in self.alert_rules.items():
            alert_rule = {
                "rule_name": rule_name,
                "metric": rule_config.get('metric'),
                "threshold": rule_config.get('threshold'),
                "comparison": rule_config.get('comparison'),
                "severity": rule_config.get('severity'),
                "action": self._determine_alert_action(rule_name, rule_config),
                "notification_config": {
                    "channels": self._get_notification_channels(),
                    "message_template": self._get_alert_message_template(rule_name),
                    "escalation_policy": self._get_escalation_policy(rule_config.get('severity'))
                }
            }
            alert_configurations.append(alert_rule)
        
        logger.info(f"Configured {len(alert_configurations)} real-time alert rules")
        return {
            "status": "success",
            "alert_rules": alert_configurations,
            "total_rules": len(alert_configurations)
        }
    
    def _determine_alert_action(self, rule_name: str, rule_config: Dict[str, Any]) -> str:
        """Determine appropriate action for alert rule"""
        severity = rule_config.get('severity', 'medium')
        
        actions = {
            'low': 'log_warning',
            'medium': 'send_notification',
            'high': 'send_notification_and_log',
            'critical': 'immediate_notification_and_escalate'
        }
        
        return actions.get(severity, 'send_notification')
    
    def _get_alert_message_template(self, rule_name: str) -> str:
        """Get alert message template for specific rule"""
        templates = {
            'accuracy_drop': "🚨 Tax RAG Accuracy Alert: Tax accuracy has dropped to {value:.3f} (threshold: {threshold}). Immediate review recommended.",
            'high_hallucination': "⚠️ Hallucination Risk Alert: Hallucination risk increased to {value:.3f} (threshold: {threshold}). Review response generation.",
            'jurisdiction_misclassification': "📍 Jurisdiction Alert: Classification accuracy is {value:.3f} (threshold: {threshold}). Check query routing.",
            'gpu_memory_warning': "💾 Memory Alert: GPU memory usage at {value}MB (threshold: {threshold}MB). Monitor system resources.",
            'response_latency': "⏱️ Performance Alert: Response latency is {value}ms (threshold: {threshold}ms). Check pipeline performance."
        }
        
        return templates.get(rule_name, "Alert: {metric} is {value} (threshold: {threshold})")
    
    def _get_escalation_policy(self, severity: str) -> Dict[str, Any]:
        """Get escalation policy based on severity"""
        policies = {
            'low': {"escalate_after": "1h", "escalate_to": "team_lead"},
            'medium': {"escalate_after": "30m", "escalate_to": "team_lead"},
            'high': {"escalate_after": "15m", "escalate_to": "senior_engineer"},
            'critical': {"escalate_after": "5m", "escalate_to": "on_call_engineer"}
        }
        
        return policies.get(severity, policies['medium'])
    
    def get_dashboard_url(self) -> str:
        """Get Phoenix dashboard URL"""
        phoenix_config = config_manager.phoenix.server if hasattr(config_manager, 'phoenix') else {}
        host = phoenix_config.get('host', 'localhost')
        port = phoenix_config.get('port', 6006)
        
        return f"http://{host}:{port}"
    
    def export_dashboard_config(self, file_path: str) -> bool:
        """Export dashboard configuration to JSON file"""
        try:
            config = self.setup_dashboard()
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Dashboard configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dashboard config: {str(e)}")
            return False
    
    def validate_dashboard_config(self) -> Dict[str, Any]:
        """Validate dashboard configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        # Check Phoenix availability
        if not PHOENIX_AVAILABLE:
            validation_results["errors"].append("Phoenix library not available")
            validation_results["valid"] = False
        
        # Validate alert rules
        for rule_name, rule_config in self.alert_rules.items():
            if not rule_config.get('metric'):
                validation_results["errors"].append(f"Alert rule '{rule_name}' missing metric")
                validation_results["valid"] = False
            
            if not rule_config.get('threshold'):
                validation_results["warnings"].append(f"Alert rule '{rule_name}' missing threshold")
        
        # Validate custom views
        for view in self.custom_views:
            if not view.get('name'):
                validation_results["errors"].append("Custom view missing name")
                validation_results["valid"] = False
        
        validation_results["summary"] = {
            "phoenix_available": PHOENIX_AVAILABLE,
            "alert_rules_count": len(self.alert_rules),
            "custom_views_count": len(self.custom_views),
            "dashboard_url": self.get_dashboard_url()
        }
        
        return validation_results


# Global dashboard instance (lazy initialization to avoid config issues)
# tax_rag_dashboard = TaxRagDashboard()