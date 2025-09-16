"""
Memory monitoring utilities for RAG Tax System
Monitors GPU and system memory usage
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics container"""
    timestamp: datetime
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_free_mb: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    system_memory_mb: float = 0.0
    system_memory_available_mb: float = 0.0
    system_memory_percent: float = 0.0
    process_memory_mb: float = 0.0


class MemoryMonitor:
    """
    Monitor GPU and system memory usage
    Provides utilities for memory management
    """
    
    def __init__(self):
        """Initialize memory monitor"""
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        
        if self.gpu_available:
            logger.info(f"GPU monitoring enabled. {self.device_count} GPU(s) detected")
            for i in range(self.device_count):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
        else:
            logger.info("GPU monitoring disabled - CUDA not available")
    
    def get_gpu_memory_info(self, device: int = 0) -> Dict[str, float]:
        """
        Get GPU memory information
        
        Args:
            device: GPU device index
            
        Returns:
            Dictionary with memory information in MB
        """
        if not self.gpu_available or device >= self.device_count:
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
                "utilization_percent": 0.0
            }
        
        try:
            # Get memory info
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            # Get GPU properties
            gpu_props = torch.cuda.get_device_properties(device)
            total_memory = gpu_props.total_memory
            
            # Calculate free memory
            free_memory = total_memory - reserved
            
            # Convert to MB
            allocated_mb = allocated / (1024 ** 2)
            reserved_mb = reserved / (1024 ** 2)
            free_mb = free_memory / (1024 ** 2)
            total_mb = total_memory / (1024 ** 2)
            
            # Calculate utilization percentage
            utilization_percent = (reserved / total_memory) * 100
            
            return {
                "allocated_mb": allocated_mb,
                "reserved_mb": reserved_mb,
                "free_mb": free_mb,
                "total_mb": total_mb,
                "utilization_percent": utilization_percent
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {str(e)}")
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
                "utilization_percent": 0.0
            }
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """
        Get system memory information
        
        Returns:
            Dictionary with system memory information in MB
        """
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info().rss
            
            return {
                "total_mb": memory.total / (1024 ** 2),
                "available_mb": memory.available / (1024 ** 2),
                "percent": memory.percent,
                "process_memory_mb": process_memory / (1024 ** 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system memory info: {str(e)}")
            return {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "percent": 0.0,
                "process_memory_mb": 0.0
            }
    
    def get_memory_stats(self, device: int = 0) -> MemoryStats:
        """
        Get comprehensive memory statistics
        
        Args:
            device: GPU device index
            
        Returns:
            MemoryStats object
        """
        gpu_info = self.get_gpu_memory_info(device)
        system_info = self.get_system_memory_info()
        
        return MemoryStats(
            timestamp=datetime.now(),
            gpu_allocated_mb=gpu_info["allocated_mb"],
            gpu_reserved_mb=gpu_info["reserved_mb"],
            gpu_free_mb=gpu_info["free_mb"],
            gpu_total_mb=gpu_info["total_mb"],
            gpu_utilization_percent=gpu_info["utilization_percent"],
            system_memory_mb=system_info["total_mb"],
            system_memory_available_mb=system_info["available_mb"],
            system_memory_percent=system_info["percent"],
            process_memory_mb=system_info["process_memory_mb"]
        )
    
    def log_memory_stats(self, device: int = 0, level: str = "INFO") -> None:
        """
        Log current memory statistics
        
        Args:
            device: GPU device index
            level: Logging level
        """
        stats = self.get_memory_stats(device)
        
        log_func = getattr(logger, level.lower())
        
        if self.gpu_available:
            log_func(
                f"Memory Stats - GPU: {stats.gpu_reserved_mb:.1f}/{stats.gpu_total_mb:.1f}MB "
                f"({stats.gpu_utilization_percent:.1f}%), "
                f"System: {stats.process_memory_mb:.1f}MB "
                f"({stats.system_memory_percent:.1f}% used)"
            )
        else:
            log_func(
                f"Memory Stats - System: {stats.process_memory_mb:.1f}MB "
                f"({stats.system_memory_percent:.1f}% used)"
            )
    
    def check_memory_threshold(self, gpu_threshold_percent: float = 90.0, device: int = 0) -> Dict[str, bool]:
        """
        Check if memory usage exceeds thresholds
        
        Args:
            gpu_threshold_percent: GPU memory threshold percentage
            device: GPU device index
            
        Returns:
            Dictionary with threshold check results
        """
        stats = self.get_memory_stats(device)
        
        return {
            "gpu_threshold_exceeded": stats.gpu_utilization_percent > gpu_threshold_percent,
            "gpu_utilization": stats.gpu_utilization_percent,
            "system_memory_percent": stats.system_memory_percent
        }
    
    def clear_gpu_cache(self, device: Optional[int] = None) -> None:
        """
        Clear GPU memory cache
        
        Args:
            device: Specific device to clear (all devices if None)
        """
        if not self.gpu_available:
            logger.warning("Cannot clear GPU cache - CUDA not available")
            return
        
        try:
            if device is not None:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                logger.info(f"Cleared GPU {device} memory cache")
            else:
                torch.cuda.empty_cache()
                logger.info("Cleared all GPU memory caches")
                
        except Exception as e:
            logger.error(f"Failed to clear GPU cache: {str(e)}")
    
    def optimize_memory(self, target_utilization: float = 80.0, device: int = 0) -> bool:
        """
        Attempt to optimize memory usage
        
        Args:
            target_utilization: Target GPU utilization percentage
            device: GPU device index
            
        Returns:
            True if optimization was successful
        """
        if not self.gpu_available:
            return False
        
        stats_before = self.get_memory_stats(device)
        
        if stats_before.gpu_utilization_percent <= target_utilization:
            logger.info(f"Memory already optimized: {stats_before.gpu_utilization_percent:.1f}%")
            return True
        
        logger.info(f"Optimizing memory from {stats_before.gpu_utilization_percent:.1f}%")
        
        # Clear cache
        self.clear_gpu_cache(device)
        
        # Check if optimization was successful
        stats_after = self.get_memory_stats(device)
        success = stats_after.gpu_utilization_percent <= target_utilization
        
        logger.info(
            f"Memory optimization {'successful' if success else 'failed'}: "
            f"{stats_after.gpu_utilization_percent:.1f}%"
        )
        
        return success
    
    def get_memory_summary(self, device: int = 0) -> str:
        """
        Get formatted memory summary string
        
        Args:
            device: GPU device index
            
        Returns:
            Formatted memory summary
        """
        stats = self.get_memory_stats(device)
        
        summary_parts = []
        
        if self.gpu_available:
            summary_parts.append(
                f"GPU Memory: {stats.gpu_reserved_mb:.0f}MB/{stats.gpu_total_mb:.0f}MB "
                f"({stats.gpu_utilization_percent:.1f}%)"
            )
        
        summary_parts.append(
            f"System Memory: {stats.process_memory_mb:.0f}MB "
            f"({stats.system_memory_percent:.1f}% system used)"
        )
        
        return " | ".join(summary_parts)


# Global memory monitor instance
memory_monitor = MemoryMonitor()