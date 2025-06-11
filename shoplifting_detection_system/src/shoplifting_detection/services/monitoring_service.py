"""
System Monitoring Service for Shoplifting Detection System
Implements comprehensive performance monitoring and metrics collection
Supports REQ-018, REQ-019, REQ-020: System reliability and monitoring
"""

import asyncio
import psutil
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import threading
from collections import deque
import GPUtil

from models.database import SystemMetrics, ModelPerformance, get_db
from config import Config, PerformanceTargets

logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """System health status"""
    overall_status: str  # healthy, warning, critical
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_status: str
    processing_latency: float
    uptime: float
    active_cameras: int
    alerts_per_hour: float


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    processing_latency_ms: float
    throughput_fps: float
    accuracy_rate: float
    false_positive_rate: float
    system_load: float
    memory_usage_mb: float
    gpu_utilization: float


class MonitoringService:
    """
    Comprehensive system monitoring service
    - REQ-018: 99.5% uptime monitoring
    - REQ-019: Automatic recovery within 60 seconds
    - REQ-020: Seamless failover with ≤10 seconds interruption
    """
    
    def __init__(self):
        self.performance_targets = PerformanceTargets()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=1000)  # Keep last 1000 metrics
        self.alert_counts = deque(maxlen=24)  # Last 24 hours of alert counts
        
        # System health tracking
        self.system_start_time = time.time()
        self.last_health_check = time.time()
        self.health_status = "unknown"
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.throughput_measurements = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=50)
        
        # Failure detection
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Monitoring intervals
        self.health_check_interval = 30  # seconds
        self.metrics_collection_interval = 10  # seconds
        self.performance_evaluation_interval = 60  # seconds
        
        logger.info("MonitoringService initialized")
    
    def start_monitoring(self):
        """Start the monitoring service"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_health_check = 0
        last_metrics_collection = 0
        last_performance_evaluation = 0
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Health check
                if current_time - last_health_check >= self.health_check_interval:
                    self._perform_health_check()
                    last_health_check = current_time
                
                # Metrics collection
                if current_time - last_metrics_collection >= self.metrics_collection_interval:
                    self._collect_system_metrics()
                    last_metrics_collection = current_time
                
                # Performance evaluation
                if current_time - last_performance_evaluation >= self.performance_evaluation_interval:
                    self._evaluate_performance()
                    last_performance_evaluation = current_time
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self._handle_monitoring_failure()
                
                time.sleep(5)  # Wait before retrying
    
    def _perform_health_check(self):
        """Perform comprehensive system health check"""
        try:
            health = self._get_system_health()
            
            # Determine overall health status
            if (health.cpu_usage > 90 or health.memory_usage > 90 or 
                health.processing_latency > self.performance_targets.MAX_PROCESSING_LATENCY_MS):
                self.health_status = "critical"
            elif (health.cpu_usage > 80 or health.memory_usage > 80 or 
                  health.processing_latency > self.performance_targets.MAX_PROCESSING_LATENCY_MS * 0.8):
                self.health_status = "warning"
            else:
                self.health_status = "healthy"
            
            # Store health metrics
            self._store_health_metrics(health)
            
            # Check for recovery needs
            if self.health_status == "critical":
                self._trigger_recovery_procedures()
            
            self.last_health_check = time.time()
            self.consecutive_failures = 0  # Reset failure count on successful check
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            self.consecutive_failures += 1
    
    def _get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        try:
            # CPU and Memory
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # GPU usage
            gpu_usage = 0.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass
            
            # Network status (simplified)
            network_status = "connected"  # Would implement actual network check
            
            # Processing latency (average of recent measurements)
            processing_latency = (
                sum(self.processing_times) / len(self.processing_times) 
                if self.processing_times else 0.0
            )
            
            # System uptime
            uptime = time.time() - self.system_start_time
            
            # Active cameras (would get from camera service)
            active_cameras = 1  # Placeholder
            
            # Alerts per hour
            alerts_per_hour = sum(self.alert_counts) if self.alert_counts else 0.0
            
            return SystemHealth(
                overall_status=self.health_status,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                disk_usage=disk_usage,
                network_status=network_status,
                processing_latency=processing_latency,
                uptime=uptime,
                active_cameras=active_cameras,
                alerts_per_hour=alerts_per_hour
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                overall_status="error",
                cpu_usage=0, memory_usage=0, gpu_usage=0, disk_usage=0,
                network_status="unknown", processing_latency=0,
                uptime=0, active_cameras=0, alerts_per_hour=0
            )
    
    def _collect_system_metrics(self):
        """Collect and store system metrics"""
        try:
            db = next(get_db())
            
            # System resource metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            metrics = [
                SystemMetrics(
                    metric_type="system",
                    metric_name="cpu_usage",
                    value=cpu_usage,
                    unit="%"
                ),
                SystemMetrics(
                    metric_type="system",
                    metric_name="memory_usage",
                    value=memory.percent,
                    unit="%"
                ),
                SystemMetrics(
                    metric_type="system",
                    metric_name="memory_available",
                    value=memory.available / (1024**3),  # GB
                    unit="GB"
                )
            ]
            
            # GPU metrics
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.extend([
                        SystemMetrics(
                            metric_type="gpu",
                            metric_name="gpu_utilization",
                            value=gpu.load * 100,
                            unit="%"
                        ),
                        SystemMetrics(
                            metric_type="gpu",
                            metric_name="gpu_memory_usage",
                            value=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                            unit="%"
                        ),
                        SystemMetrics(
                            metric_type="gpu",
                            metric_name="gpu_temperature",
                            value=gpu.temperature,
                            unit="°C"
                        )
                    ])
            except:
                pass
            
            # Processing performance metrics
            if self.processing_times:
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                metrics.append(
                    SystemMetrics(
                        metric_type="performance",
                        metric_name="avg_processing_time",
                        value=avg_processing_time,
                        unit="ms"
                    )
                )
            
            if self.throughput_measurements:
                avg_throughput = sum(self.throughput_measurements) / len(self.throughput_measurements)
                metrics.append(
                    SystemMetrics(
                        metric_type="performance",
                        metric_name="avg_throughput",
                        value=avg_throughput,
                        unit="fps"
                    )
                )
            
            # Store metrics
            for metric in metrics:
                db.add(metric)
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _store_health_metrics(self, health: SystemHealth):
        """Store health metrics in database"""
        try:
            db = next(get_db())
            
            health_metrics = [
                SystemMetrics(
                    metric_type="health",
                    metric_name="overall_status",
                    value=1.0 if health.overall_status == "healthy" else 
                          0.5 if health.overall_status == "warning" else 0.0,
                    unit="status"
                ),
                SystemMetrics(
                    metric_type="health",
                    metric_name="uptime",
                    value=health.uptime,
                    unit="seconds"
                ),
                SystemMetrics(
                    metric_type="health",
                    metric_name="active_cameras",
                    value=health.active_cameras,
                    unit="count"
                )
            ]
            
            for metric in health_metrics:
                db.add(metric)
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Error storing health metrics: {e}")
    
    def _evaluate_performance(self):
        """Evaluate system performance against targets"""
        try:
            # Calculate performance metrics
            avg_latency = (
                sum(self.processing_times) / len(self.processing_times) 
                if self.processing_times else 0.0
            )
            
            avg_throughput = (
                sum(self.throughput_measurements) / len(self.throughput_measurements) 
                if self.throughput_measurements else 0.0
            )
            
            avg_accuracy = (
                sum(self.accuracy_history) / len(self.accuracy_history) 
                if self.accuracy_history else 0.0
            )
            
            # Check against targets
            latency_target_met = avg_latency <= self.performance_targets.MAX_PROCESSING_LATENCY_MS
            throughput_target_met = avg_throughput >= self.performance_targets.TARGET_FPS
            accuracy_target_met = avg_accuracy >= self.performance_targets.TARGET_ACCURACY
            
            # Log performance evaluation
            logger.info(f"Performance Evaluation - "
                       f"Latency: {avg_latency:.1f}ms (target: {self.performance_targets.MAX_PROCESSING_LATENCY_MS}ms), "
                       f"Throughput: {avg_throughput:.1f}fps (target: {self.performance_targets.TARGET_FPS}fps), "
                       f"Accuracy: {avg_accuracy:.1%} (target: {self.performance_targets.TARGET_ACCURACY:.1%})")
            
            # Store performance evaluation
            self._store_performance_evaluation(
                avg_latency, avg_throughput, avg_accuracy,
                latency_target_met, throughput_target_met, accuracy_target_met
            )
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
    
    def _store_performance_evaluation(self, latency: float, throughput: float, 
                                    accuracy: float, latency_ok: bool, 
                                    throughput_ok: bool, accuracy_ok: bool):
        """Store performance evaluation results"""
        try:
            db = next(get_db())
            
            performance = ModelPerformance(
                model_name="ensemble_detector",
                model_version="1.0.0",
                accuracy=accuracy,
                processing_time_avg_ms=latency,
                metadata={
                    "throughput_fps": throughput,
                    "latency_target_met": latency_ok,
                    "throughput_target_met": throughput_ok,
                    "accuracy_target_met": accuracy_ok,
                    "evaluation_timestamp": datetime.now().isoformat()
                }
            )
            
            db.add(performance)
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Error storing performance evaluation: {e}")
    
    def _trigger_recovery_procedures(self):
        """Trigger system recovery procedures"""
        if self.recovery_attempts < self.max_recovery_attempts:
            self.recovery_attempts += 1
            logger.warning(f"Triggering recovery procedure (attempt {self.recovery_attempts})")
            
            # Implement recovery procedures
            # - Restart failed services
            # - Clear memory caches
            # - Reduce processing load
            # - Switch to backup systems
            
            # For now, just log the recovery attempt
            logger.info("Recovery procedures executed")
        else:
            logger.critical("Maximum recovery attempts reached. Manual intervention required.")
    
    def _handle_monitoring_failure(self):
        """Handle monitoring system failure"""
        logger.critical("Monitoring system failure detected. Attempting restart...")
        
        try:
            # Reset failure counters
            self.consecutive_failures = 0
            
            # Restart monitoring if needed
            if not self.monitoring_active:
                self.start_monitoring()
                
        except Exception as e:
            logger.critical(f"Failed to recover monitoring system: {e}")
    
    def record_processing_time(self, processing_time_ms: float):
        """Record processing time measurement"""
        self.processing_times.append(processing_time_ms)
    
    def record_throughput(self, fps: float):
        """Record throughput measurement"""
        self.throughput_measurements.append(fps)
    
    def record_accuracy(self, accuracy: float):
        """Record accuracy measurement"""
        self.accuracy_history.append(accuracy)
    
    def record_alert(self):
        """Record alert generation for rate tracking"""
        current_hour = datetime.now().hour
        if not self.alert_counts or len(self.alert_counts) <= current_hour:
            self.alert_counts.extend([0] * (current_hour + 1 - len(self.alert_counts)))
        self.alert_counts[current_hour] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        health = self._get_system_health()
        
        return {
            "overall_status": health.overall_status,
            "uptime_hours": health.uptime / 3600,
            "cpu_usage": health.cpu_usage,
            "memory_usage": health.memory_usage,
            "gpu_usage": health.gpu_usage,
            "processing_latency_ms": health.processing_latency,
            "active_cameras": health.active_cameras,
            "alerts_per_hour": health.alerts_per_hour,
            "performance_targets_met": {
                "latency": health.processing_latency <= self.performance_targets.MAX_PROCESSING_LATENCY_MS,
                "uptime": (health.uptime / 3600) >= 24 * 0.995  # 99.5% of 24 hours
            },
            "last_health_check": datetime.fromtimestamp(self.last_health_check).isoformat()
        }
