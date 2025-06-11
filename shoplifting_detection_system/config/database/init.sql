-- Create tables for shoplifting detection system

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    person_id INTEGER,
    x_coordinate INTEGER,
    y_coordinate INTEGER,
    width INTEGER,
    height INTEGER,
    description TEXT,
    image_path VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES events(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    message TEXT NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS person_tracks (
    id SERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    x_coordinate INTEGER NOT NULL,
    y_coordinate INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    confidence FLOAT NOT NULL
);

CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX idx_person_tracks_person_id ON person_tracks(person_id);
CREATE INDEX idx_person_tracks_timestamp ON person_tracks(timestamp);
