import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class AlertSeverity(Enum):
    CRITICAL = "critical"  # 90-100% confidence
    HIGH = "high"         # 75-89% confidence
    MEDIUM = "medium"     # 60-74% confidence
    LOW = "low"          # 45-59% confidence


class DetectionBehavior(Enum):
    ITEM_CONCEALMENT = "item_concealment"
    SECURITY_TAG_REMOVAL = "security_tag_removal"
    POCKET_STUFFING = "pocket_stuffing"
    BAG_LOADING = "bag_loading"
    COORDINATED_THEFT = "coordinated_theft"
    PRICE_TAG_SWITCHING = "price_tag_switching"
    EXIT_WITHOUT_PAYMENT = "exit_without_payment"
    MULTIPLE_ITEM_HANDLING = "multiple_item_handling"


@dataclass
class PerformanceTargets:
    """Performance targets as per requirements REQ-012, REQ-013, REQ-015"""
    TARGET_ACCURACY: float = 0.95  # ≥95% true positive rate
    MAX_FALSE_POSITIVE_RATE: float = 0.02  # ≤2% per hour
    MAX_PROCESSING_LATENCY_MS: int = 200  # ≤200ms end-to-end
    TARGET_FPS: int = 30  # 30 FPS processing
    MAX_FRAME_DROPS: float = 0.05  # ≤5% frame drops
    MAX_CONCURRENT_FEEDS: int = 32  # Support 32+ camera feeds
    TARGET_UPTIME: float = 0.995  # 99.5% availability


@dataclass
class SecurityConfig:
    """Security configuration as per REQ-035"""
    ENCRYPTION_ALGORITHM: str = "AES-256"
    TLS_VERSION: str = "1.3"
    USE_HSM: bool = True
    ZERO_TRUST_NETWORK: bool = True
    AUDIT_LOGGING: bool = True


class Config:
    # Database Configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://shoplifter_user:shoplifter_pass@localhost:5432/shoplifter_db")

    # Camera Configuration
    CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", 0))
    MAX_CAMERA_FEEDS = int(os.getenv("MAX_CAMERA_FEEDS", 32))
    CAMERA_RESOLUTION = os.getenv("CAMERA_RESOLUTION", "1920x1080")
    CAMERA_FPS = int(os.getenv("CAMERA_FPS", 30))

    # Model Configuration
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "models/pose_estimation.pt")
    ACTION_MODEL_PATH = os.getenv(
        "ACTION_MODEL_PATH", "models/action_recognition.pt")
    REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", "models/person_reid.pt")
    ANOMALY_MODEL_PATH = os.getenv(
        "ANOMALY_MODEL_PATH", "models/anomaly_detection.pkl")

    # Performance Configuration
    PERFORMANCE = PerformanceTargets()

    # Security Configuration
    SECURITY = SecurityConfig()

    # Detection Thresholds (REQ-002)
    CRITICAL_THRESHOLD = 0.90  # 90-100% confidence
    HIGH_THRESHOLD = 0.75      # 75-89% confidence
    MEDIUM_THRESHOLD = 0.60    # 60-74% confidence
    LOW_THRESHOLD = 0.45       # 45-59% confidence

    # Legacy thresholds (maintained for backward compatibility)
    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.75))
    MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")  # Legacy compatibility
    CROUCHING_THRESHOLD = 0.6
    HAND_MOVEMENT_THRESHOLD = 50
    LOITERING_TIME_THRESHOLD = 30
    ALERT_COOLDOWN = 10

    # Enhanced Detection Parameters
    CONCEALMENT_DETECTION_THRESHOLD = 0.7
    TAG_REMOVAL_THRESHOLD = 0.8
    COORDINATION_THRESHOLD = 0.75
    ITEM_INTERACTION_THRESHOLD = 0.6

    # Tracking Parameters
    MAX_TRACKING_DISTANCE = 100
    TRACK_TIMEOUT = 5
    REID_THRESHOLD = 0.7  # Person re-identification threshold

    # Data Retention (REQ-009, REQ-010)
    VIDEO_RETENTION_DAYS = int(os.getenv("VIDEO_RETENTION_DAYS", 90))
    ANALYTICS_RETENTION_DAYS = int(
        os.getenv("ANALYTICS_RETENTION_DAYS", 730))  # 2 years

    # Processing Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
    GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", 0.8))

    # Monitoring and Logging
    ENABLE_PROMETHEUS = os.getenv(
        "ENABLE_PROMETHEUS", "true").lower() == "true"
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8080))
    API_WORKERS = int(os.getenv("API_WORKERS", 4))

    # Microservices Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    KAFKA_BOOTSTRAP_SERVERS = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    # Debug and Development
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENABLE_TESTING_MODE = os.getenv(
        "ENABLE_TESTING_MODE", "False").lower() == "true"

    # Compliance and Privacy (REQ-037)
    GDPR_COMPLIANCE = True
    CCPA_COMPLIANCE = True
    DATA_ANONYMIZATION = True
    CONSENT_MANAGEMENT = True
