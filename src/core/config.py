"""
Configuration management for RAG Tax System
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    """Configuration for models"""
    name: str
    cache_dir: str
    device: str = "cuda"


@dataclass
class EmbeddingConfig(ModelConfig):
    """Configuration for embedding model"""
    max_seq_length: int = 512
    batch_size: int = 32


@dataclass
class LLMConfig(ModelConfig):
    """Configuration for language model"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    type: str = "chromadb"
    persist_directory: str = "./data/vector_db"
    collection_name: str = "tax_documents"
    distance_metric: str = "cosine"


@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing"""
    chunk_size: int = 512
    chunk_overlap: int = 128
    pdf_path: str = "./resources/information_material/BusinessTaxBasics_0.pdf"
    processed_docs_path: str = "./data/processed"


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 2048
    system_prompt: str = ""


@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    max_gpu_memory_gb: int = 10
    clear_cache_interval: int = 10
    monitor_memory: bool = True


@dataclass
class RouterConfig:
    """Configuration for query router"""
    confidence_threshold: float = 0.6
    fallback_to_general: bool = True
    enable_pattern_matching: bool = True
    enable_llm_extraction: bool = True
    database_path: str = "./resources/customers_data/customers.db"
    connection_timeout: int = 30
    include_tax_bracket: bool = True
    include_filing_status: bool = True
    include_state_info: bool = True
    max_context_enhancement: int = 200
    # Smart fallback settings
    enable_smart_fallback: bool = True
    confidence_threshold_for_fallback: float = 0.75
    log_fallback_decisions: bool = True
    show_fallback_message: bool = True


class ConfigManager:
    """Manages system configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Find the project root directory (where config.yaml is located)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up from src/core/ to project root
        self.config_path = project_root / config_path
        self.config_data = self._load_config()
        
        # Initialize configuration objects
        self.embedding = self._create_embedding_config()
        self.llm = self._create_llm_config()
        self.vector_store = self._create_vector_store_config()
        self.document_processing = self._create_document_processing_config()
        self.rag = self._create_rag_config()
        self.memory = self._create_memory_config()
        self.router = self._create_router_config()
        
        # Phoenix monitoring configuration (as dict for flexibility)
        self.phoenix = self._get_phoenix_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_embedding_config(self) -> EmbeddingConfig:
        """Create embedding configuration"""
        model_config = self.config_data["models"]["embedding"]
        project_root = self.config_path.parent
        return EmbeddingConfig(
            name=model_config["name"],
            cache_dir=str(project_root / model_config["cache_dir"]),
            max_seq_length=model_config["max_seq_length"],
            batch_size=model_config["batch_size"],
            device=model_config["device"]
        )
    
    def _create_llm_config(self) -> LLMConfig:
        """Create LLM configuration"""
        model_config = self.config_data["models"]["llm"]
        project_root = self.config_path.parent
        return LLMConfig(
            name=model_config["name"],
            cache_dir=str(project_root / model_config["cache_dir"]),
            max_new_tokens=model_config["max_new_tokens"],
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            device=model_config["device"]
        )
    
    def _create_vector_store_config(self) -> VectorStoreConfig:
        """Create vector store configuration"""
        vs_config = self.config_data["vector_store"]
        project_root = self.config_path.parent
        return VectorStoreConfig(
            type=vs_config["type"],
            persist_directory=str(project_root / vs_config["persist_directory"]),
            collection_name=vs_config["collection_name"],
            distance_metric=vs_config["distance_metric"]
        )
    
    def _create_document_processing_config(self) -> DocumentProcessingConfig:
        """Create document processing configuration"""
        doc_config = self.config_data["document_processing"]
        project_root = self.config_path.parent
        return DocumentProcessingConfig(
            chunk_size=doc_config["chunk_size"],
            chunk_overlap=doc_config["chunk_overlap"],
            pdf_path=str(project_root / doc_config["pdf_path"]),
            processed_docs_path=str(project_root / doc_config["processed_docs_path"])
        )
    
    def _create_rag_config(self) -> RAGConfig:
        """Create RAG configuration"""
        rag_config = self.config_data["rag"]
        return RAGConfig(
            top_k=rag_config["retrieval"]["top_k"],
            similarity_threshold=rag_config["retrieval"]["similarity_threshold"],
            max_context_length=rag_config["generation"]["max_context_length"],
            system_prompt=rag_config["generation"]["system_prompt"]
        )
    
    def _create_memory_config(self) -> MemoryConfig:
        """Create memory configuration"""
        mem_config = self.config_data["memory"]
        return MemoryConfig(
            max_gpu_memory_gb=mem_config["max_gpu_memory_gb"],
            clear_cache_interval=mem_config["clear_cache_interval"],
            monitor_memory=mem_config["monitor_memory"]
        )
    
    def _create_router_config(self) -> RouterConfig:
        """Create router configuration"""
        if "router" not in self.config_data:
            # Return default config if router section is missing
            return RouterConfig()
        
        router_config = self.config_data["router"]
        project_root = self.config_path.parent
        
        return RouterConfig(
            confidence_threshold=router_config.get("classification", {}).get("confidence_threshold", 0.6),
            fallback_to_general=router_config.get("classification", {}).get("fallback_to_general", True),
            enable_pattern_matching=router_config.get("user_identification", {}).get("enable_pattern_matching", True),
            enable_llm_extraction=router_config.get("user_identification", {}).get("enable_llm_extraction", True),
            database_path=str(project_root / router_config.get("database", {}).get("path", "./resources/customers_data/customers.db")),
            connection_timeout=router_config.get("database", {}).get("connection_timeout", 30),
            include_tax_bracket=router_config.get("personalization", {}).get("include_tax_bracket", True),
            include_filing_status=router_config.get("personalization", {}).get("include_filing_status", True),
            include_state_info=router_config.get("personalization", {}).get("include_state_info", True),
            max_context_enhancement=router_config.get("personalization", {}).get("max_context_enhancement", 200),
            # Smart fallback settings
            enable_smart_fallback=router_config.get("fallback", {}).get("enable_smart_fallback", True),
            confidence_threshold_for_fallback=router_config.get("fallback", {}).get("confidence_threshold_for_fallback", 0.75),
            log_fallback_decisions=router_config.get("fallback", {}).get("log_fallback_decisions", True),
            show_fallback_message=router_config.get("fallback", {}).get("show_fallback_message", True)
        )
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get configuration value by path (e.g., 'models.embedding.name')"""
        keys = path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def _get_phoenix_config(self) -> Dict[str, Any]:
        """Get Phoenix monitoring configuration as flexible dict"""
        phoenix_config = self.config_data.get("phoenix", {})
        
        # Create nested attribute access using SimpleNamespace-like behavior
        class PhoenixConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, PhoenixConfig(value))
                    else:
                        setattr(self, key, value)
        
        return PhoenixConfig(phoenix_config) if phoenix_config else None
    
    def update_config_value(self, path: str, value: Any) -> None:
        """Update configuration value by path"""
        keys = path.split('.')
        config = self.config_data
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)


# Global configuration instance
config_manager = ConfigManager()