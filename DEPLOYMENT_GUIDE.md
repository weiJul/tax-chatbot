# Production Deployment Guide

## Overview

This guide covers deploying the Advanced RAG Tax System in production environments with full monitoring, evaluation, and MCP server capabilities.

## Prerequisites

### Hardware Requirements
- **GPU**: GeForce RTX 2080 Ti (11GB VRAM) or equivalent CUDA-compatible GPU
- **CPU**: Multi-core processor (8+ cores recommended for concurrent users)
- **RAM**: 32GB+ system RAM for production workloads
- **Storage**: 50GB+ SSD for models, data, and logging
- **Network**: Stable internet connection for model downloads

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), Windows 10+, or macOS
- **Python**: 3.10 or 3.11 (tested with 3.10)
- **CUDA**: 11.8+ with compatible drivers
- **Docker**: Optional, for containerized deployment

## Installation & Setup

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd tax-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python test_environment.py
```

### 3. Document Processing

```bash
# Process documents with jurisdiction metadata (required for first deployment)
python process_documents_with_jurisdiction.py

# Verify document processing
python -c "from src.core.vector_store import vector_store; vector_store.connect(); result = vector_store.collection.get(include=['metadatas']); jurisdictions = {}; [jurisdictions.update({meta.get('jurisdiction', 'missing'): jurisdictions.get(meta.get('jurisdiction', 'missing'), 0) + 1}) for meta in result['metadatas']]; print(f'Total: {len(result[\"metadatas\"])}, Breakdown: {jurisdictions}')"
```

### 4. Database Setup

```bash
# Create user database for MCP integration
cd resources/customers_data
python create_database.py
cd ../..

# Verify database creation
python -c "from src.core.tax_mcp_client import mcp_server; status = mcp_server.check_database_connection(); print(f'Database Status: {status[\"status\"]} - {status[\"user_count\"]} users')"
```

## Configuration

### 1. Production Configuration

Create `config.production.yaml` based on `config.yaml`:

```yaml
# Production-specific overrides
models:
  llm:
    temperature: 0.3  # More conservative for production
    max_new_tokens: 150  # Shorter responses for efficiency

memory:
  clear_cache_interval: 10  # More frequent cache clearing
  monitor_memory: true

# Phoenix monitoring (essential for production)
phoenix:
  tracing:
    enabled: true
  evaluation:
    enabled: true
    auto_eval_interval: 5  # More frequent evaluation
  alerts:
    enabled: true

# Logging
logging:
  level: "INFO"
  file: "/var/log/tax-rag/system.log"  # Adjust path for your system
```

### 2. Environment Variables

```bash
# Create .env file
cat > .env << EOF
ENVIRONMENT=production
CUDA_VISIBLE_DEVICES=0
TRANSFORMERS_CACHE=./models/cache
HF_HOME=./models/huggingface
LOG_LEVEL=INFO
EOF
```

## Deployment Options

### Option 1: Systemd Service (Linux)

#### 1. Create Service Files

**CLI Service:**
```bash
sudo tee /etc/systemd/system/tax-rag-cli.service > /dev/null << EOF
[Unit]
Description=Tax RAG CLI Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/tax-rag
Environment=PATH=/path/to/tax-rag/venv/bin
ExecStart=/path/to/tax-rag/venv/bin/python src/interfaces/cli_chat.py --headless
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tax-rag-cli

[Install]
WantedBy=multi-user.target
EOF
```

**Web Service:**
```bash
sudo tee /etc/systemd/system/tax-rag-web.service > /dev/null << EOF
[Unit]
Description=Tax RAG Web Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/tax-rag
Environment=PATH=/path/to/tax-rag/venv/bin
ExecStart=/path/to/tax-rag/venv/bin/streamlit run src/interfaces/web_interface.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tax-rag-web

[Install]
WantedBy=multi-user.target
EOF
```

**MCP Server Service:**
```bash
sudo tee /etc/systemd/system/tax-rag-mcp.service > /dev/null << EOF
[Unit]
Description=Tax RAG MCP Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/tax-rag
Environment=PATH=/path/to/tax-rag/venv/bin
ExecStart=/path/to/tax-rag/venv/bin/python mcp_tax_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tax-rag-mcp

[Install]
WantedBy=multi-user.target
EOF
```

#### 2. Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable tax-rag-web tax-rag-mcp

# Start services
sudo systemctl start tax-rag-web tax-rag-mcp

# Check status
sudo systemctl status tax-rag-web tax-rag-mcp
```

### Option 2: Docker Deployment

#### 1. Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA (adjust version as needed)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-11-8

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models

# Expose ports
EXPOSE 8501 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "src/interfaces/web_interface.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

#### 2. Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  tax-rag-web:
    build: .
    ports:
      - "8501:8501"
      - "6006:6006"  # Phoenix monitoring
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./resources:/app/resources
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - ENVIRONMENT=production
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  tax-rag-mcp:
    build: .
    command: ["python", "mcp_tax_server.py"]
    volumes:
      - ./data:/app/data
      - ./resources:/app/resources
    environment:
      - ENVIRONMENT=production
    restart: unless-stopped

volumes:
  data:
  models:
  logs:
```

#### 3. Deploy with Docker

```bash
# Build and start services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f tax-rag-web
```

## Monitoring & Evaluation Setup

### 1. Phoenix-Arize AI Setup

```bash
# Start Phoenix server in background
python start_phoenix_server.py &

# Verify Phoenix is running
curl -f http://localhost:6006/health || echo "Phoenix not accessible"

# Access Phoenix dashboard
# Open browser to http://localhost:6006
```

### 2. Enable Automated Evaluation

```bash
# Test evaluation system
python -c "from src.evaluation.automated_evaluation import AutomatedEvaluationService; service = AutomatedEvaluationService(); print('Evaluation system ready')"

# Start background evaluation (if using standalone)
nohup python -c "
from src.evaluation.automated_evaluation import AutomatedEvaluationService
import time
service = AutomatedEvaluationService()
service.start_scheduled_evaluations()
while True:
    time.sleep(60)
" > logs/evaluation.log 2>&1 &
```

## Load Balancing & Scaling

### 1. Nginx Configuration

```nginx
# /etc/nginx/sites-available/tax-rag
upstream tax_rag_backend {
    server 127.0.0.1:8501;
    # Add more instances for horizontal scaling
    # server 127.0.0.1:8502;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://tax_rag_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Phoenix monitoring (optional external access)
    location /monitoring/ {
        proxy_pass http://127.0.0.1:6006/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        auth_basic "Monitoring Access";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
```

### 2. Enable and Configure Nginx

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/tax-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Create monitoring auth (optional)
sudo htpasswd -c /etc/nginx/.htpasswd monitor
```

## Security Considerations

### 1. Firewall Configuration

```bash
# Allow necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8501/tcp  # Streamlit (if direct access needed)

# Block Phoenix from external access (optional)
sudo ufw deny 6006/tcp

# Enable firewall
sudo ufw enable
```

### 2. Data Security

- **Database Encryption**: Ensure `customers.db` has proper file permissions (`600`)
- **Log Rotation**: Configure logrotate for application logs
- **SSL/TLS**: Use Let's Encrypt or similar for HTTPS in production
- **API Keys**: Store any external API keys in environment variables, not config files

### 3. Access Control

```bash
# Restrict file permissions
chmod 600 resources/customers_data/customers.db
chmod 700 data/
chmod -R 755 src/

# Create dedicated user
sudo useradd -m -s /bin/bash tax-rag-user
sudo chown -R tax-rag-user:tax-rag-user /path/to/tax-rag
```

## Backup & Recovery

### 1. Database Backup

```bash
#!/bin/bash
# backup_database.sh
DB_PATH="./resources/customers_data/customers.db"
BACKUP_DIR="./backups/database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
cp $DB_PATH "$BACKUP_DIR/customers_$TIMESTAMP.db"

# Keep only last 30 days of backups
find $BACKUP_DIR -name "customers_*.db" -mtime +30 -delete

echo "Database backed up to $BACKUP_DIR/customers_$TIMESTAMP.db"
```

### 2. Vector Database Backup

```bash
#!/bin/bash
# backup_vectordb.sh
VECTOR_DIR="./data/vector_db"
BACKUP_DIR="./backups/vectordb"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf "$BACKUP_DIR/vectordb_$TIMESTAMP.tar.gz" -C $VECTOR_DIR .

# Keep only last 7 days of vector DB backups (larger files)
find $BACKUP_DIR -name "vectordb_*.tar.gz" -mtime +7 -delete

echo "Vector database backed up to $BACKUP_DIR/vectordb_$TIMESTAMP.tar.gz"
```

### 3. Automated Backup with Cron

```bash
# Add to crontab (crontab -e)
# Daily database backup at 2 AM
0 2 * * * /path/to/tax-rag/backup_database.sh

# Weekly vector database backup at 3 AM on Sundays
0 3 * * 0 /path/to/tax-rag/backup_vectordb.sh
```

## Performance Optimization

### 1. GPU Memory Management

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check system memory usage
python -c "from src.utils.memory_monitor import memory_monitor; print(memory_monitor.get_memory_summary())"
```

### 2. Batch Processing Optimization

```yaml
# Optimize config.yaml for production load
models:
  embedding:
    batch_size: 16  # Reduce if memory issues occur
memory:
  clear_cache_interval: 5  # More frequent clearing under load
  max_gpu_memory_gb: 9  # Conservative limit
```

### 3. Database Optimization

```sql
-- Run on customers.db for better performance
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_tax_id ON customers(tax_id);
CREATE INDEX IF NOT EXISTS idx_customers_name ON customers(first_name, last_name);
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Check GPU memory usage
nvidia-smi

# Clear cache manually
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Reduce batch size in config.yaml
```

#### 2. Phoenix Server Not Starting
```bash
# Check port availability
netstat -ln | grep 6006

# Kill existing Phoenix processes
pkill -f phoenix

# Restart with different port
python start_phoenix_server.py --port 6007
```

#### 3. MCP Server Connection Issues
```bash
# Test MCP server manually
python -c "from src.core.tax_mcp_client import mcp_server; print(mcp_server.check_database_connection())"

# Check database permissions
ls -la resources/customers_data/customers.db

# Restart MCP server
sudo systemctl restart tax-rag-mcp
```

### Log Analysis

```bash
# System logs
sudo journalctl -u tax-rag-web -f

# Application logs
tail -f data/rag_system.log

# Phoenix monitoring logs
tail -f data/phoenix_monitoring.log

# Evaluation logs
tail -f logs/evaluation.log
```

## Maintenance

### Regular Tasks

1. **Weekly**: Review Phoenix monitoring dashboards
2. **Monthly**: Analyze evaluation reports and accuracy metrics
3. **Quarterly**: Update dependencies and models if needed
4. **As needed**: Add new users to database, update documents

### Health Checks

```bash
#!/bin/bash
# health_check.sh
echo "=== Tax RAG System Health Check ==="

# Check services
systemctl is-active tax-rag-web tax-rag-mcp

# Check disk space
df -h | grep -E "(data|models|logs)"

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Check system components
python -c "
from src.core.llama_retrieval import hierarchical_retrieval
from src.core.tax_mcp_client import mcp_server
print('✅ Hierarchical retrieval ready')
status = mcp_server.check_database_connection()
print(f'✅ MCP Server: {status[\"status\"]} - {status[\"user_count\"]} users')
"

echo "=== Health Check Complete ==="
```

### Updates & Patches

```bash
# Update dependencies (test in staging first)
pip install --upgrade -r requirements.txt

# Update models (if needed)
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5', force_download=True)
AutoModel.from_pretrained('BAAI/bge-base-en-v1.5', force_download=True)
"

# Test system after updates
python test_environment.py
python test_router.py
python test_hierarchical_retrieval.py
```

---

## Production Checklist

Before going live:

- [ ] All dependencies installed and tested
- [ ] Documents processed and vector database populated
- [ ] User database created with sample data
- [ ] Phoenix monitoring configured and running
- [ ] Automated evaluation enabled
- [ ] SSL/HTTPS configured (if applicable)
- [ ] Firewall and security measures in place
- [ ] Backup scripts configured and tested
- [ ] Health monitoring and alerting set up
- [ ] Load testing completed
- [ ] Documentation updated with deployment-specific details

**Note**: This system is designed for educational and research purposes. Ensure compliance with all relevant tax authority guidelines and data protection regulations when deploying with real tax data.