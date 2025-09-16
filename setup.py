#!/usr/bin/env python3
"""
Setup script for RAG Tax System
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name="advanced-rag-tax-system",
    version="2.0.0",
    description="Production-ready jurisdiction-aware tax assistant with LlamaIndex hierarchical retrieval, intelligent routing, and Phoenix-Arize AI monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tax RAG System",
    author_email="",
    url="",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Requirements
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
        "monitoring": [
            "arize-phoenix>=4.0.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
        ],
        "mcp": [
            "mcp[cli]>=0.1.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ]
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "tax-rag-cli=interfaces.cli_chat:main",
            "tax-rag-web=interfaces.web_interface:main",
            "tax-rag-mcp=mcp_tax_server:main",
            "tax-rag-test=test_environment:main",
            "tax-rag-phoenix=start_phoenix_server:main",
        ]
    },
    
    # MCP Server configuration
    "mcp_servers": {
        "tax-database": {
            "command": "python",
            "args": ["mcp_tax_server.py"],
            "description": "Tax database access via Model Context Protocol"
        }
    },
    
    # Package data
    package_data={
        "": [
            "*.yaml",
            "*.yml", 
            "*.json",
            "*.txt",
            "*.md"
        ],
    },
    
    # Include additional files
    include_package_data=True,
    
    # Classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Office/Business :: Financial",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Internet :: WWW/HTTP :: WSGI :: Application",
        "Framework :: FastAPI",
        "Environment :: Web Environment",
    ],
    
    # Keywords
    keywords=[
        "rag", "retrieval-augmented-generation", "llm", "nlp", "tax", 
        "question-answering", "chromadb", "bge-embeddings", "pytorch", "cuda",
        "llamaindex", "hierarchical-retrieval", "jurisdiction-aware", "langchain",
        "query-router", "mcp", "model-context-protocol", "phoenix-arize", 
        "ai-monitoring", "evaluation", "observability", "tax-assistant"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "",
        "Source": "",
        "Tracker": "",
    },
    
    # Zip safety
    zip_safe=False,
)