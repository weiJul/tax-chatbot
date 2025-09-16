# Phoenix-Arize AI Monitoring & Evaluation Guide

## Overview

The Tax RAG System includes comprehensive **Phoenix-Arize AI monitoring** for production-ready observability, evaluation, and performance tracking. This system provides automated LLM evaluation, prompt performance analysis, and real-time monitoring dashboards.

## Architecture

### Phoenix Integration Components

```
Tax RAG System ←→ Phoenix Tracer ←→ Phoenix Server ←→ Dashboard
                        ↓
                  Evaluation Engine
                        ↓
                  Automated Reports
```

### Key Features
- **Real-time Tracing**: Every query traced with detailed performance metrics
- **Automated Evaluation**: Scheduled accuracy, hallucination, and jurisdiction testing
- **Custom Dashboards**: Tax-specific performance views and metrics
- **Production Monitoring**: Comprehensive observability for deployed systems
- **Alert System**: Proactive notifications for performance degradation

## Phoenix Server Setup

### 1. Installation & Dependencies

Phoenix is already included in `requirements.txt`:
```bash
# Install Phoenix and dependencies
pip install arize-phoenix>=4.0.0
pip install opentelemetry-api>=1.20.0
pip install opentelemetry-sdk>=1.20.0
```

### 2. Start Phoenix Server

#### Option 1: Manual Start
```bash
# Start Phoenix server manually
python start_phoenix_server.py

# Check if server is running
curl -f http://localhost:6006/health || echo "Phoenix not accessible"
```

#### Option 2: Integrated Start (Recommended)
Phoenix automatically starts when the Tax RAG system initializes:

```bash
# Phoenix starts automatically with these commands
python src/interfaces/cli_chat.py
# or
streamlit run src/interfaces/web_interface.py
```

### 3. Phoenix Configuration

Configuration is managed through `config.yaml`:

```yaml
phoenix:
  # Server configuration
  server:
    host: "localhost"
    port: 6006
    grpc_port: 6007
    launch_app: true
    notebook: false
  
  # Tracing configuration  
  tracing:
    enabled: true
    project_name: "tax-rag-system"
    endpoint: "http://localhost:6006/v1/traces"
    service_name: "tax-rag-assistant"
    service_version: "1.0.0"
  
  # Evaluation configuration
  evaluation:
    enabled: true
    auto_eval_interval: 10  # Run evaluations every 10 queries
    tax_accuracy_threshold: 0.85
    hallucination_threshold: 0.1
```

## Monitoring Dashboards

### 1. Access Phoenix Dashboard

Open your browser to: **http://localhost:6006**

### 2. Custom Tax RAG Dashboards

The system includes pre-configured custom views:

#### RAG Pipeline Performance
- **Metrics**: `retrieval_duration_ms`, `generation_duration_ms`, `pipeline_success_rate`
- **Purpose**: Monitor end-to-end pipeline performance
- **Thresholds**: 
  - Retrieval: < 500ms (target)
  - Generation: < 2000ms (target)
  - Success Rate: > 95%

#### Tax Accuracy Metrics
- **Metrics**: `tax_accuracy_score`, `jurisdiction_accuracy`, `citation_accuracy`
- **Purpose**: Track tax-specific response quality
- **Thresholds**:
  - Tax Accuracy: > 85%
  - Jurisdiction Accuracy: > 90%
  - Citation Accuracy: > 80%

#### User Experience
- **Metrics**: `query_classification_accuracy`, `user_lookup_success_rate`
- **Purpose**: Monitor router and user identification performance
- **Thresholds**:
  - Classification Accuracy: > 80%
  - User Lookup Success: > 95%

#### System Health
- **Metrics**: `memory_gpu_usage`, `embedding_batch_size`, `retrieval_cache_hit_rate`
- **Purpose**: Monitor resource utilization and efficiency
- **Thresholds**:
  - GPU Memory: < 9GB (RTX 2080 Ti)
  - Cache Hit Rate: > 70%

### 3. Real-time Monitoring

#### Live Trace View
1. Navigate to **Traces** tab in Phoenix dashboard
2. Filter by `project_name: "tax-rag-system"`
3. Monitor real-time query processing

#### Performance Metrics
- **Query Volume**: Requests per minute/hour
- **Response Times**: P50, P95, P99 latencies
- **Error Rates**: Failed queries and error types
- **Resource Usage**: GPU memory, CPU utilization

## Automated Evaluation System

### 1. Evaluation Components

The system includes specialized evaluators:

#### Tax Accuracy Evaluator
- **Purpose**: Validates factual accuracy of tax-related responses
- **Method**: Compares responses against ground truth tax knowledge
- **Metrics**: Accuracy score (0.0-1.0), fact verification
- **Threshold**: 85% accuracy required

#### Hallucination Detector
- **Purpose**: Identifies fabricated or incorrect information
- **Method**: Cross-references responses with source documents
- **Metrics**: Hallucination rate (0.0-1.0), confidence scores
- **Threshold**: < 10% hallucination rate

#### Jurisdiction Classification Evaluator
- **Purpose**: Validates jurisdiction detection accuracy
- **Method**: Tests California vs general query classification
- **Metrics**: Classification accuracy, precision, recall
- **Threshold**: 90% jurisdiction accuracy

### 2. Evaluation Datasets

Located in `src/evaluation/datasets/`:

#### Tax Accuracy Dataset
```json
{
  "queries": [
    {
      "question": "What is B&O tax?",
      "expected_answer": "Business and Occupation tax",
      "jurisdiction": "general",
      "difficulty": "easy"
    }
  ]
}
```

#### Hallucination Prompts
```json
{
  "prompts": [
    {
      "question": "What are the 2025 tax rates?",
      "expected_behavior": "should_refuse_or_clarify",
      "reason": "future_information"
    }
  ]
}
```

#### Jurisdiction Test Cases
```json
{
  "test_cases": [
    {
      "query": "California FTB requirements",
      "expected_jurisdiction": "california",
      "confidence_threshold": 0.8
    }
  ]
}
```

### 3. Evaluation Schedule

Configure automated evaluation in `config.yaml`:

```yaml
phoenix:
  evaluation:
    schedule:
      daily_comprehensive: "02:00"  # Daily full evaluation at 2 AM
      hourly_quick_check: "every_hour"  # Quick accuracy checks
      real_time_sampling: 0.1  # Evaluate 10% of queries in real-time
```

### 4. Running Evaluations

#### Manual Evaluation
```bash
# Run comprehensive evaluation
python -c "
from src.evaluation.automated_evaluation import AutomatedEvaluationService
service = AutomatedEvaluationService()
results = service.run_comprehensive_evaluation()
print(f'Overall Score: {results[\"overall_score\"]:.2f}')
"

# Run specific evaluator
python -c "
from src.evaluation.tax_evaluators import TaxAccuracyEvaluator
evaluator = TaxAccuracyEvaluator()
score = evaluator.evaluate_query('What is B&O tax?', 'Business and Occupation tax...')
print(f'Tax Accuracy: {score:.2f}')
"
```

#### Scheduled Evaluation
```bash
# Start automated evaluation service
python -c "
from src.evaluation.automated_evaluation import AutomatedEvaluationService
import time
service = AutomatedEvaluationService()
service.start_scheduled_evaluations()
print('Evaluation service started - running in background')
while True:
    time.sleep(60)  # Keep service running
"
```

## Alert System

### 1. Alert Configuration

Configure alerts in `config.yaml`:

```yaml
phoenix:
  alerts:
    enabled: true
    channels:
      console: true
      log_file: true
      # email: false  # Configure for production
      # slack: false  # Configure for production
    
    rules:
      accuracy_drop:
        metric: "tax_accuracy_score"
        threshold: 0.80
        comparison: "below"
        severity: "high"
      
      high_hallucination:
        metric: "hallucination_rate"
        threshold: 0.15
        comparison: "above"
        severity: "critical"
```

### 2. Alert Types

#### Performance Alerts
- **Response Latency**: When queries take > 5 seconds
- **GPU Memory**: When usage > 9GB on RTX 2080 Ti
- **Error Rate**: When > 5% of queries fail

#### Quality Alerts
- **Accuracy Drop**: Tax accuracy < 80%
- **High Hallucination**: Hallucination rate > 15%
- **Jurisdiction Misclassification**: Classification accuracy < 90%

#### System Alerts
- **Database Connection**: MCP server connectivity issues
- **Phoenix Server**: Monitoring service unavailable
- **Model Loading**: Embedding/LLM loading failures

### 3. Alert Handling

#### Console Alerts
```bash
[2025-09-16 14:30:15] ALERT [HIGH]: Tax accuracy dropped to 0.78 (threshold: 0.80)
[2025-09-16 14:30:15] ALERT [CRITICAL]: Hallucination rate increased to 0.18 (threshold: 0.15)
```

#### Log File Alerts
Located in `data/phoenix_monitoring.log`:
```
2025-09-16 14:30:15,123 - phoenix_tracer - WARNING - Alert triggered: accuracy_drop
2025-09-16 14:30:15,124 - phoenix_tracer - CRITICAL - Alert triggered: high_hallucination
```

## Production Monitoring Setup

### 1. Dashboard Configuration

#### Enable External Access (Production)
```nginx
# Nginx configuration for Phoenix dashboard
server {
    listen 443 ssl;
    server_name monitoring.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:6006;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Basic authentication
        auth_basic "Phoenix Monitoring";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

#### Create Monitoring User
```bash
# Create monitoring access credentials
sudo htpasswd -c /etc/nginx/.htpasswd monitoring
```

### 2. Data Retention

Configure data retention in Phoenix:

```yaml
phoenix:
  dashboard:
    retention_days: 30  # Keep 30 days of traces
    refresh_interval: 5  # Update every 5 seconds
```

### 3. Backup & Export

#### Export Evaluation Results
```bash
# Export evaluation data
python -c "
from src.evaluation.automated_evaluation import AutomatedEvaluationService
service = AutomatedEvaluationService()
service.export_evaluation_history('./backups/evaluation_export.json')
print('Evaluation data exported')
"
```

#### Backup Phoenix Data
```bash
# Backup Phoenix traces (if using persistent storage)
tar -czf backups/phoenix_traces_$(date +%Y%m%d).tar.gz data/phoenix/
```

## Performance Analysis

### 1. Key Metrics to Monitor

#### Response Time Analysis
```python
# Analyze query performance
import pandas as pd
from src.monitoring.phoenix_dashboard import TaxRagDashboard

dashboard = TaxRagDashboard()
metrics = dashboard.get_performance_metrics(hours=24)

print(f"Average Response Time: {metrics['avg_response_time']:.2f}ms")
print(f"P95 Response Time: {metrics['p95_response_time']:.2f}ms")
print(f"Query Success Rate: {metrics['success_rate']:.1%}")
```

#### Accuracy Trends
```python
# Track accuracy over time
from src.evaluation.automated_evaluation import AutomatedEvaluationService

service = AutomatedEvaluationService()
trend = service.get_accuracy_trend(days=7)

print("7-day accuracy trend:")
for date, score in trend.items():
    print(f"  {date}: {score:.2f}")
```

### 2. Optimization Recommendations

#### Based on Phoenix Data
1. **Slow Queries**: Identify queries with > 3s response time
2. **High Memory Usage**: Monitor GPU memory spikes
3. **Low Cache Hit Rate**: Optimize embedding caching
4. **Jurisdiction Misclassification**: Improve detection keywords

#### Performance Tuning
```python
# Get performance recommendations
from src.monitoring.phoenix_dashboard import TaxRagDashboard

dashboard = TaxRagDashboard()
recommendations = dashboard.get_optimization_recommendations()

for category, suggestions in recommendations.items():
    print(f"{category}:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
```

## Troubleshooting

### 1. Phoenix Server Issues

#### Server Not Starting
```bash
# Check port availability
netstat -ln | grep 6006

# Kill existing processes
pkill -f phoenix

# Start with different port
python start_phoenix_server.py --port 6007
```

#### Dashboard Not Accessible
```bash
# Verify server is running
curl -f http://localhost:6006/health

# Check firewall
sudo ufw status | grep 6006

# View server logs
tail -f data/phoenix_monitoring.log
```

### 2. Evaluation Issues

#### Evaluation Not Running
```bash
# Check evaluation service status
python -c "
from src.evaluation.automated_evaluation import AutomatedEvaluationService
service = AutomatedEvaluationService()
status = service.get_evaluation_status()
print(f'Evaluation Status: {status}')
"

# Restart evaluation service
sudo systemctl restart tax-rag-evaluation  # If using systemd
```

#### Low Evaluation Scores
```bash
# Debug specific evaluator
python -c "
from src.evaluation.tax_evaluators import TaxAccuracyEvaluator
evaluator = TaxAccuracyEvaluator()
evaluator.debug_mode = True
# Run test evaluation to see detailed results
"
```

### 3. Memory and Performance Issues

#### High Memory Usage
```bash
# Monitor Phoenix memory usage
ps aux | grep phoenix
nvidia-smi  # Check GPU memory

# Reduce trace retention
# Edit config.yaml: phoenix.dashboard.retention_days: 7
```

#### Slow Dashboard Loading
```bash
# Clear old traces
python -c "
from src.monitoring.phoenix_dashboard import TaxRagDashboard
dashboard = TaxRagDashboard()
dashboard.cleanup_old_traces(days=7)
print('Old traces cleaned up')
"
```

## Best Practices

### 1. Production Deployment
- **Enable HTTPS**: Use SSL certificates for dashboard access
- **Authentication**: Implement proper user authentication
- **Network Security**: Restrict Phoenix port access via firewall
- **Data Privacy**: Ensure trace data doesn't contain sensitive user information

### 2. Monitoring Strategy
- **Start Simple**: Begin with basic accuracy and performance monitoring
- **Gradual Enhancement**: Add custom evaluators based on specific needs
- **Regular Review**: Weekly review of evaluation results and trends
- **Alert Tuning**: Adjust thresholds based on actual system performance

### 3. Evaluation Schedule
- **Real-time**: 10% sample rate for immediate feedback
- **Hourly**: Quick accuracy checks on recent queries
- **Daily**: Comprehensive evaluation at low-traffic hours
- **Weekly**: Full system performance analysis

### 4. Data Management
- **Retention Policy**: Keep 30 days of detailed traces, longer for summaries
- **Export Strategy**: Regular exports of evaluation results
- **Storage Optimization**: Compress and archive old monitoring data

---

## Quick Start Checklist

### Initial Setup
- [ ] Phoenix dependencies installed
- [ ] Phoenix server started and accessible
- [ ] Basic dashboards configured
- [ ] Evaluation datasets prepared

### Production Readiness
- [ ] Automated evaluation enabled
- [ ] Alert system configured
- [ ] Dashboard authentication set up
- [ ] Data retention policy implemented
- [ ] Backup procedures established

### Ongoing Monitoring
- [ ] Daily evaluation reports reviewed
- [ ] Performance trends analyzed
- [ ] Alert thresholds tuned
- [ ] System optimizations applied

This comprehensive monitoring system ensures production-ready observability and continuous improvement of the Tax RAG system.