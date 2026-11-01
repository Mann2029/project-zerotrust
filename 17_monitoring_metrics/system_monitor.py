import os
import time
import psutil
from typing import Dict, Any

class SystemMonitor:
    """Monitors hardware health, AI inference latencies, and Prometheus telemetry."""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_pipeline_latency_ms = 0.0
        self.total_verifications = 0
        print("[INFO] System & AI Performance Monitor initialized.")

    def record_pipeline_execution(self, latency_ms: float):
        """Records AI pipeline inference durations for throughput calculations."""
        self.request_count += 1
        self.total_verifications += 1
        self.total_pipeline_latency_ms += latency_ms

    def get_system_health(self) -> Dict[str, Any]:
        """Collects live RAM, CPU, disk, and process telemetry."""
        process = psutil.Process(os.getpid())
        ram_info = psutil.virtual_memory()
        uptime_seconds = time.time() - self.start_time

        avg_latency = (
            self.total_pipeline_latency_ms / self.total_verifications
            if self.total_verifications > 0
            else 0.0
        )

        return {
            "status": "HEALTHY",
            "uptime_seconds": round(uptime_seconds, 2),
            "cpu_utilization_percent": psutil.cpu_percent(interval=None),
            "system_ram_used_percent": ram_info.percent,
            "monolith_process_ram_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "total_verifications_processed": self.total_verifications,
            "avg_pipeline_latency_ms": round(avg_latency, 2)
        }

    def generate_prometheus_metrics(self) -> str:
        """Formats performance metrics into Prometheus scrape format."""
        health = self.get_system_health()
        metrics = [
            f"# HELP attendance_verifications_total Total number of face verifications processed.",
            f"# TYPE attendance_verifications_total counter",
            f"attendance_verifications_total {health['total_verifications_processed']}",
            f"# HELP attendance_avg_pipeline_latency_ms Average AI verification latency in milliseconds.",
            f"# TYPE attendance_avg_pipeline_latency_ms gauge",
            f"attendance_avg_pipeline_latency_ms {health['avg_pipeline_latency_ms']}",
            f"# HELP process_memory_usage_mb Monolith resident memory usage in megabytes.",
            f"# TYPE process_memory_usage_mb gauge",
            f"process_memory_usage_mb {health['monolith_process_ram_mb']}",
            f"# HELP system_cpu_utilization_percent System CPU usage percent.",
            f"# TYPE system_cpu_utilization_percent gauge",
            f"system_cpu_utilization_percent {health['cpu_utilization_percent']}"
        ]
        return "\n".join(metrics) + "\n"