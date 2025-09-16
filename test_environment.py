#!/usr/bin/env python3
"""
Environment Test Script for RAG Tax System
Tests all dependencies and system requirements
"""
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentTester:
    """Tests environment setup and dependencies"""
    
    def __init__(self):
        """Initialize environment tester"""
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def print_header(self, title: str) -> None:
        """Print formatted section header"""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def print_result(self, test_name: str, status: str, details: str = "") -> None:
        """Print formatted test result"""
        status_icons = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️",
            "INFO": "ℹ️"
        }
        
        icon = status_icons.get(status, "❓")
        print(f"{icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
    
    def test_python_version(self) -> bool:
        """Test Python version compatibility"""
        try:
            version = sys.version_info
            version_str = f"{version.major}.{version.minor}.{version.micro}"
            
            if version.major == 3 and version.minor >= 8:
                self.print_result("Python Version", "PASS", f"Python {version_str}")
                return True
            else:
                self.print_result("Python Version", "FAIL", f"Python {version_str} (need >= 3.8)")
                self.errors.append(f"Python version {version_str} is too old (need >= 3.8)")
                return False
                
        except Exception as e:
            self.print_result("Python Version", "FAIL", str(e))
            self.errors.append(f"Python version check failed: {str(e)}")
            return False
    
    def test_gpu_availability(self) -> bool:
        """Test GPU and CUDA availability"""
        try:
            import torch
            
            # Test basic PyTorch installation
            torch_version = torch.__version__
            self.print_result("PyTorch Installation", "PASS", f"Version {torch_version}")
            
            # Test CUDA availability
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                cuda_version = torch.version.cuda
                
                self.print_result("CUDA Availability", "PASS", f"CUDA {cuda_version}")
                self.print_result("GPU Detection", "PASS", f"{device_count} GPU(s) found")
                self.print_result("Current GPU", "PASS", f"GPU {current_device}: {device_name}")
                
                # Test GPU memory
                gpu_props = torch.cuda.get_device_properties(current_device)
                total_memory_gb = gpu_props.total_memory / (1024**3)
                self.print_result("GPU Memory", "PASS", f"{total_memory_gb:.1f} GB")
                
                if total_memory_gb >= 10:
                    self.print_result("Memory Check", "PASS", "Sufficient for RAG system")
                else:
                    self.print_result("Memory Check", "WARN", "May be insufficient for large models")
                    self.warnings.append(f"GPU has only {total_memory_gb:.1f}GB memory")
                
                return True
            else:
                self.print_result("CUDA Availability", "FAIL", "CUDA not available")
                self.errors.append("CUDA is not available - GPU acceleration required")
                return False
                
        except ImportError:
            self.print_result("PyTorch Installation", "FAIL", "PyTorch not installed")
            self.errors.append("PyTorch is not installed")
            return False
        except Exception as e:
            self.print_result("GPU Test", "FAIL", str(e))
            self.errors.append(f"GPU test failed: {str(e)}")
            return False
    
    def test_core_dependencies(self) -> bool:
        """Test core ML dependencies"""
        dependencies = {
            "transformers": "Hugging Face Transformers",
            "sentence_transformers": "Sentence Transformers",
            "chromadb": "ChromaDB Vector Database",
            "pypdf": "PDF Processing",
            "numpy": "NumPy",
            "pandas": "Pandas"
        }
        
        all_passed = True
        
        for package, description in dependencies.items():
            try:
                __import__(package)
                self.print_result(description, "PASS", f"{package} installed")
            except ImportError:
                self.print_result(description, "FAIL", f"{package} not installed")
                self.errors.append(f"{package} is not installed")
                all_passed = False
            except Exception as e:
                self.print_result(description, "FAIL", f"{package}: {str(e)}")
                self.errors.append(f"{package} import failed: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_optional_dependencies(self) -> bool:
        """Test optional dependencies"""
        optional_deps = {
            "streamlit": "Streamlit Web Interface",
            "accelerate": "Model Acceleration",
            "psutil": "System Monitoring"
        }
        
        for package, description in optional_deps.items():
            try:
                __import__(package)
                self.print_result(description, "PASS", f"{package} installed")
            except ImportError:
                self.print_result(description, "WARN", f"{package} not installed (optional)")
                self.warnings.append(f"{package} not installed (optional feature)")
            except Exception as e:
                self.print_result(description, "WARN", f"{package}: {str(e)}")
                self.warnings.append(f"{package} import issue: {str(e)}")
        
        return True  # Optional dependencies don't fail the test
    
    def test_file_structure(self) -> bool:
        """Test project file structure"""
        required_files = [
            "config.yaml",
            "requirements.txt",
            "src/core/config.py",
            "src/core/embeddings.py",
            "src/core/vector_store.py",
            "src/core/document_processor.py",
            "src/core/llm_service.py",
            "src/utils/memory_monitor.py",
            "src/utils/text_utils.py",
            "src/interfaces/cli_chat.py",
            "src/interfaces/web_interface.py",
            "resources/information_material/BusinessTaxBasics_0.pdf"
        ]
        
        all_exist = True
        
        for file_path in required_files:
            if Path(file_path).exists():
                self.print_result(f"File: {file_path}", "PASS", "Found")
            else:
                self.print_result(f"File: {file_path}", "FAIL", "Missing")
                self.errors.append(f"Required file missing: {file_path}")
                all_exist = False
        
        # Check directories
        required_dirs = ["data", "models", "src/core", "src/utils", "src/interfaces"]
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                self.print_result(f"Directory: {dir_path}", "PASS", "Found")
            else:
                self.print_result(f"Directory: {dir_path}", "FAIL", "Missing")
                self.errors.append(f"Required directory missing: {dir_path}")
                all_exist = False
        
        return all_exist
    
    def test_memory_requirements(self) -> bool:
        """Test system memory requirements"""
        try:
            import psutil
            
            # System memory
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            self.print_result("System Memory", "PASS", f"{total_gb:.1f} GB total, {available_gb:.1f} GB available")
            
            if available_gb >= 8:
                self.print_result("Memory Check", "PASS", "Sufficient system memory")
            else:
                self.print_result("Memory Check", "WARN", "Low available memory")
                self.warnings.append(f"Only {available_gb:.1f}GB system memory available")
            
            # Disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            
            self.print_result("Disk Space", "PASS", f"{free_gb:.1f} GB free")
            
            if free_gb >= 10:
                self.print_result("Disk Check", "PASS", "Sufficient disk space")
            else:
                self.print_result("Disk Check", "WARN", "Low disk space")
                self.warnings.append(f"Only {free_gb:.1f}GB disk space available")
            
            return True
            
        except ImportError:
            self.print_result("Memory Test", "WARN", "psutil not available")
            return True
        except Exception as e:
            self.print_result("Memory Test", "FAIL", str(e))
            return False
    
    def test_model_loading(self) -> bool:
        """Test if models can be loaded (quick test)"""
        try:
            from transformers import AutoTokenizer
            
            # Test tokenizer loading (lighter than full model)
            tokenizer_name = "BAAI/bge-base-en-v1.5"
            self.print_result("Model Loading Test", "INFO", f"Testing {tokenizer_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.print_result("BGE Tokenizer", "PASS", "Loaded successfully")
            
            return True
            
        except Exception as e:
            self.print_result("Model Loading Test", "WARN", f"Could not test model loading: {str(e)}")
            self.warnings.append(f"Model loading test failed: {str(e)}")
            return True  # Don't fail on this test
    
    def run_all_tests(self) -> bool:
        """Run all environment tests"""
        print("🏛️  RAG Tax System - Environment Test")
        print("Testing system requirements and dependencies...")
        
        tests = [
            ("Python Version", self.test_python_version),
            ("GPU & CUDA", self.test_gpu_availability),
            ("Core Dependencies", self.test_core_dependencies),
            ("Optional Dependencies", self.test_optional_dependencies),
            ("File Structure", self.test_file_structure),
            ("Memory Requirements", self.test_memory_requirements),
            ("Model Loading", self.test_model_loading)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            self.print_header(test_name)
            try:
                result = test_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.print_result(test_name, "FAIL", f"Test crashed: {str(e)}")
                self.errors.append(f"{test_name} test crashed: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def print_summary(self) -> None:
        """Print test summary"""
        self.print_header("Test Summary")
        
        if not self.errors and not self.warnings:
            print("🎉 All tests passed! Your environment is ready for RAG Tax System.")
        elif self.errors:
            print(f"❌ {len(self.errors)} critical error(s) found:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n🔧 Please fix the errors above before proceeding.")
            print("💡 Run 'pip install -r requirements.txt' to install missing dependencies.")
        else:
            print(f"\n✅ Environment is ready! You can proceed with implementation.")
            print("🚀 Run 'python src/interfaces/cli_chat.py' to start the CLI interface.")


def main():
    """Main entry point"""
    tester = EnvironmentTester()
    
    try:
        success = tester.run_all_tests()
        tester.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()