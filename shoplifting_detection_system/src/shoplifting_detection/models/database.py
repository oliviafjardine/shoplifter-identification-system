from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, LargeBinary, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime, timedelta
import uuid
from config import Config, AlertSeverity, DetectionBehavior

Base = declarative_base()


class Camera(Base):
    """Camera configuration and status tracking"""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200))
    ip_address = Column(String(45))  # Support IPv6
    port = Column(Integer)
    username = Column(String(100))
    password_hash = Column(String(255))  # Encrypted password
    stream_url = Column(String(500))
    resolution = Column(String(20), default="1920x1080")
    fps = Column(Integer, default=30)
    # active, inactive, maintenance, error
    status = Column(String(20), default="active")
    last_heartbeat = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    events = relationship("Event", back_populates="camera")


class Person(Base):
    """Person tracking and re-identification"""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    person_uuid = Column(UUID(as_uuid=True),
                         default=uuid.uuid4, unique=True, index=True)
    first_seen = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now())
    reid_features = Column(LargeBinary)  # Encoded re-identification features
    total_detections = Column(Integer, default=1)
    confidence_avg = Column(Float, default=0.0)
    # active, suspicious, cleared
    status = Column(String(20), default="active")

    # Relationships
    tracks = relationship("PersonTrack", back_populates="person")
    events = relationship("Event", back_populates="person")


class Event(Base):
    """Enhanced event tracking with comprehensive metadata"""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_uuid = Column(UUID(as_uuid=True),
                        default=uuid.uuid4, unique=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    event_type = Column(String(50), nullable=False, index=True)
    behavior_type = Column(String(50))  # DetectionBehavior enum values
    confidence = Column(Float, nullable=False)
    severity = Column(String(20), default="medium",
                      index=True)  # AlertSeverity enum values

    # Location and tracking
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    person_id = Column(Integer, ForeignKey("persons.id"))
    x_coordinate = Column(Integer)
    y_coordinate = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)

    # Enhanced metadata
    description = Column(Text)
    metadata = Column(JSONB)  # Store additional structured data
    video_clip_path = Column(String(500))  # Path to 30-second video clip
    image_path = Column(String(500))  # Path to key frame image
    # Path to complete evidence package
    evidence_package_path = Column(String(500))

    # Processing information
    processing_time_ms = Column(Float)  # Time taken to process this event
    model_version = Column(String(50))  # Version of ML model used

    # Relationships
    camera = relationship("Camera", back_populates="events")
    person = relationship("Person", back_populates="events")
    alerts = relationship("Alert", back_populates="event")

    # Indexes for performance
    __table_args__ = (
        Index('idx_event_timestamp_severity', 'timestamp', 'severity'),
        Index('idx_event_camera_timestamp', 'camera_id', 'timestamp'),
        Index('idx_event_person_timestamp', 'person_id', 'timestamp'),
    )


class Alert(Base):
    """Enhanced alert system with comprehensive tracking"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_uuid = Column(UUID(as_uuid=True),
                        default=uuid.uuid4, unique=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    timestamp = Column(DateTime, default=func.now(), index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium", index=True)
    priority_score = Column(Float, default=0.5)  # 0-1 priority scoring

    # Alert content
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    recommendation = Column(Text)  # Recommended action

    # Status tracking
    # active, acknowledged, resolved, false_positive
    status = Column(String(20), default="active", index=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    resolved_at = Column(DateTime)
    resolved_by = Column(String(100))
    resolution_notes = Column(Text)

    # Notification tracking
    notifications_sent = Column(JSONB)  # Track which notifications were sent
    escalation_level = Column(Integer, default=0)  # Escalation level
    escalated_at = Column(DateTime)

    # Relationships
    event = relationship("Event", back_populates="alerts")

    # Indexes
    __table_args__ = (
        Index('idx_alert_timestamp_severity', 'timestamp', 'severity'),
        Index('idx_alert_status_timestamp', 'status', 'timestamp'),
    )


class PersonTrack(Base):
    """Enhanced person tracking with trajectory analysis"""
    __tablename__ = "person_tracks"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, default=func.now(), index=True)

    # Bounding box
    x_coordinate = Column(Integer, nullable=False)
    y_coordinate = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)

    # Enhanced tracking data
    velocity_x = Column(Float, default=0.0)
    velocity_y = Column(Float, default=0.0)
    pose_keypoints = Column(JSONB)  # Pose estimation keypoints
    action_features = Column(LargeBinary)  # Action recognition features

    # Relationships
    person = relationship("Person", back_populates="tracks")
    camera = relationship("Camera")

    # Indexes
    __table_args__ = (
        Index('idx_track_person_timestamp', 'person_id', 'timestamp'),
        Index('idx_track_camera_timestamp', 'camera_id', 'timestamp'),
    )


class SystemMetrics(Base):
    """System performance and monitoring metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    # cpu, memory, gpu, processing_time, etc.
    metric_type = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # %, ms, MB, etc.
    # Optional camera-specific metrics
    camera_id = Column(Integer, ForeignKey("cameras.id"))

    # Relationships
    camera = relationship("Camera")

    # Indexes
    __table_args__ = (
        Index('idx_metrics_timestamp_type', 'timestamp', 'metric_type'),
        Index('idx_metrics_camera_timestamp', 'camera_id', 'timestamp'),
    )


class ModelPerformance(Base):
    """ML model performance tracking"""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)

    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    false_positive_rate = Column(Float)
    processing_time_avg_ms = Column(Float)

    # Test dataset information
    test_samples = Column(Integer)
    test_period_start = Column(DateTime)
    test_period_end = Column(DateTime)

    # Additional metadata
    metadata = Column(JSONB)

    # Indexes
    __table_args__ = (
        Index('idx_model_perf_timestamp', 'timestamp'),
        Index('idx_model_perf_name_version', 'model_name', 'model_version'),
    )


class AuditLog(Base):
    """Comprehensive audit logging for compliance (REQ-036)"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    user_id = Column(String(100), index=True)
    session_id = Column(String(100))
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50))  # alert, event, camera, etc.
    resource_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(String(500))

    # Action details
    action_details = Column(JSONB)
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    # Indexes
    __table_args__ = (
        Index('idx_audit_timestamp_user', 'timestamp', 'user_id'),
        Index('idx_audit_action_timestamp', 'action', 'timestamp'),
    )


class DataRetentionPolicy(Base):
    """Data retention policy tracking (REQ-009, REQ-010)"""
    __tablename__ = "data_retention_policies"

    id = Column(Integer, primary_key=True, index=True)
    # video, events, analytics, etc.
    data_type = Column(String(50), nullable=False, unique=True)
    retention_days = Column(Integer, nullable=False)
    auto_cleanup = Column(Boolean, default=True)
    last_cleanup = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# Database connection
engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)


def init_default_data():
    """Initialize default data retention policies and system configuration"""
    db = SessionLocal()
    try:
        # Check if retention policies exist
        existing_policies = db.query(DataRetentionPolicy).count()
        if existing_policies == 0:
            # Create default retention policies
            policies = [
                DataRetentionPolicy(
                    data_type="video", retention_days=Config.VIDEO_RETENTION_DAYS),
                DataRetentionPolicy(
                    data_type="events", retention_days=Config.ANALYTICS_RETENTION_DAYS),
                DataRetentionPolicy(
                    data_type="alerts", retention_days=Config.ANALYTICS_RETENTION_DAYS),
                DataRetentionPolicy(
                    data_type="person_tracks", retention_days=Config.ANALYTICS_RETENTION_DAYS),
                DataRetentionPolicy(data_type="system_metrics",
                                    retention_days=365),  # 1 year
                # 7 years for compliance
                DataRetentionPolicy(data_type="audit_logs",
                                    retention_days=2555),
            ]

            for policy in policies:
                db.add(policy)

            db.commit()
            print("Default data retention policies created")

    except Exception as e:
        print(f"Error initializing default data: {e}")
        db.rollback()
    finally:
        db.close()
