import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
import json
from datetime import datetime
import io
import base64
from typing import List, Dict

# Import our modules
from config import Config
from models.database import create_tables, get_db, Alert, Event
from services.camera_service import CameraService
from services.alert_service import AlertService
from detection.object_detector import ObjectDetector
from detection.tracker import PersonTracker
from detection.shoplifting_detector import ShopliftingDetector
from detection.anomaly_detector import AnomalyDetector

# Initialize FastAPI app
app = FastAPI(title="Shoplifting Detection System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global services
camera_service = None
alert_service = AlertService()
object_detector = ObjectDetector()
person_tracker = PersonTracker()
shoplifting_detector = ShopliftingDetector()
shoplifting_detector.set_object_detector(object_detector)
anomaly_detector = AnomalyDetector()

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

# Global variables for real-time data
current_people_count = 0
current_detections = []
iteration_count = 0


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global camera_service

    # Create database tables
    create_tables()

    # Initialize camera service
    camera_service = CameraService()

    # Register alert callback for WebSocket broadcasting
    async def alert_callback(alert_data: Dict):
        message = {
            "type": "alert",
            "data": alert_data
        }
        await manager.broadcast(json.dumps(message))

    alert_service.register_alert_callback(alert_callback)

    # Start camera capture
    if camera_service.start_capture():
        print("Camera service started successfully")
        # Start processing loop
        asyncio.create_task(process_video_stream())
    else:
        print("Failed to start camera service")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera_service
    if camera_service:
        camera_service.stop_capture()


async def process_video_stream():
    """Main video processing loop"""
    global current_people_count, current_detections

    while True:
        try:
            if not camera_service or not camera_service.is_running:
                await asyncio.sleep(1)
                continue

            # Get latest frame
            frame_data = await camera_service.get_frame_async()
            if not frame_data:
                await asyncio.sleep(0.1)
                continue

            frame = frame_data['frame']
            timestamp = frame_data['timestamp']

            # Detect people in frame
            people_detections = object_detector.detect_people(frame)

            # Update person tracking (this handles deduplication and unique person counting)
            tracked_people = person_tracker.update_tracks(people_detections)

            # Update global people count based on unique tracked people
            # Count unique active tracks
            new_people_count = len(person_tracker.tracks)
            current_detections = people_detections

            # Debug logging (only when count changes or there are detections)
            if Config.DEBUG and (len(people_detections) > 0 or new_people_count != current_people_count):
                print(
                    f"DEBUG: Raw detections: {len(people_detections)}, Active tracks: {new_people_count}, Tracked people: {len(tracked_people)}")

            # Broadcast people count update if changed
            if new_people_count != current_people_count:
                current_people_count = new_people_count
                print(
                    f"DEBUG: People count changed from {current_people_count} to {new_people_count}")
                detection_update = {
                    "type": "detection_update",
                    "data": {
                        "people_count": current_people_count,
                        "unique_tracks": len(person_tracker.tracks),
                        "raw_detections": len(people_detections),
                        "timestamp": timestamp.isoformat()
                    }
                }
                await manager.broadcast(json.dumps(detection_update))

            if tracked_people:

                # Analyze behavior for each tracked person
                for person_data in tracked_people:
                    behavior_analysis = shoplifting_detector.analyze_person_behavior(
                        person_data, frame)

                    # Update anomaly detector
                    anomaly_detector.update_model(
                        behavior_analysis, person_data['track_data'])

                    # Check for anomalies
                    anomaly_result = anomaly_detector.detect_anomaly(
                        behavior_analysis, person_data['track_data'])

                    # Create alert if suspicious behavior detected
                    if (shoplifting_detector.is_behavior_suspicious(behavior_analysis) or
                            anomaly_result.get('is_anomaly', False)):

                        event_data = {
                            'x_coordinate': person_data['bbox']['center_x'],
                            'y_coordinate': person_data['bbox']['center_y'],
                            'width': person_data['bbox']['width'],
                            'height': person_data['bbox']['height'],
                            'image_path': None  # Could save frame here
                        }

                        await alert_service.create_alert(event_data, behavior_analysis, anomaly_result)

            # Periodic cleanup of old person states (every 100 iterations)
            global iteration_count
            iteration_count += 1

            if iteration_count % 100 == 0:
                shoplifting_detector.cleanup_old_states()

            # Small delay to prevent excessive CPU usage
            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in video processing loop: {e}")
            await asyncio.sleep(1)


@app.get("/")
async def get_dashboard():
    """Serve the main dashboard"""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/video_feed")
async def video_feed():
    """Stream video feed with detections"""
    def generate_frames():
        while True:
            if not camera_service or not camera_service.is_running:
                break

            frame = camera_service.get_current_frame()
            if frame is None:
                continue

            # Get detections and draw them
            detections = object_detector.detect_objects(frame)
            annotated_frame = object_detector.draw_detections(
                frame, detections)

            # Encode frame
            frame_bytes = camera_service.encode_frame_to_jpeg(annotated_frame)
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "acknowledge_alert":
                alert_id = message.get("alert_id")
                if alert_id:
                    success = await alert_service.acknowledge_alert(alert_id)
                    response = {
                        "type": "alert_acknowledged",
                        "alert_id": alert_id,
                        "success": success
                    }
                    await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    alerts = await alert_service.get_recent_alerts(limit)
    return {"alerts": alerts}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    alert_stats = alert_service.get_alert_stats()
    camera_info = camera_service.get_camera_info() if camera_service else {}
    anomaly_stats = anomaly_detector.get_model_stats()

    return {
        "alerts": alert_stats,
        "camera": camera_info,
        "anomaly_detection": anomaly_stats,
        "tracking": {
            "active_tracks": len(person_tracker.tracks)
        }
    }


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Acknowledge an alert"""
    success = await alert_service.acknowledge_alert(alert_id)
    if success:
        return {"message": "Alert acknowledged successfully"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/api/camera/info")
async def get_camera_info():
    """Get camera information"""
    if camera_service:
        return camera_service.get_camera_info()
    else:
        raise HTTPException(
            status_code=503, detail="Camera service not available")


@app.get("/api/detections")
async def get_current_detections():
    """Get current detection data"""
    global current_people_count, current_detections

    # Ensure person_tracker is available
    if not person_tracker:
        return {
            "people_count": 0,
            "detections": [],
            "active_tracks": 0,
            "raw_detections": 0
        }

    return {
        "people_count": len(person_tracker.tracks),  # Unique tracked people
        "detections": current_detections,
        "active_tracks": len(person_tracker.tracks),
        "raw_detections": len(current_detections) if current_detections else 0
    }


@app.post("/api/clear-tracks")
async def clear_tracks():
    """Clear all person tracks - for debugging"""
    global current_people_count, person_tracker

    if person_tracker:
        person_tracker.clear_all_tracks()
        current_people_count = 0

        # Broadcast the update
        detection_update = {
            "type": "detection_update",
            "data": {
                "people_count": 0,
                "unique_tracks": 0,
                "raw_detections": 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        await manager.broadcast(json.dumps(detection_update))

        return {"message": "All tracks cleared", "people_count": 0}

    return {"message": "Person tracker not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
