from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from config import Config

Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    event_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    person_id = Column(Integer)
    x_coordinate = Column(Integer)
    y_coordinate = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    description = Column(Text)
    image_path = Column(String(255))
    
    alerts = relationship("Alert", back_populates="event")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    timestamp = Column(DateTime, default=func.now())
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")
    message = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    
    event = relationship("Event", back_populates="alerts")

class PersonTrack(Base):
    __tablename__ = "person_tracks"
    
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    x_coordinate = Column(Integer, nullable=False)
    y_coordinate = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)

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
