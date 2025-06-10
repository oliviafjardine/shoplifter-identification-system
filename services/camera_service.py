import cv2
import numpy as np
import asyncio
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import threading
import queue
from config import Config

class CameraService:
    def __init__(self, camera_source: int = None):
        self.camera_source = camera_source or Config.CAMERA_SOURCE
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.frame_callbacks = []
        self.capture_thread = None
        
    def start_capture(self) -> bool:
        """
        Start camera capture
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_source}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            print(f"Camera capture started on source {self.camera_source}")
            return True
            
        except Exception as e:
            print(f"Error starting camera capture: {e}")
            return False
    
    def stop_capture(self):
        """
        Stop camera capture
        """
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Camera capture stopped")
    
    def _capture_loop(self):
        """
        Main capture loop running in separate thread
        """
        while self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame,
                        'timestamp': datetime.now()
                    })
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait({
                            'frame': frame,
                            'timestamp': datetime.now()
                        })
                    except queue.Empty:
                        pass
                
                # Call registered callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        print(f"Error in frame callback: {e}")
                
                # Small delay to prevent excessive CPU usage
                cv2.waitKey(1)
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                break
    
    def get_latest_frame(self) -> Optional[Dict]:
        """
        Get the latest frame from the queue
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame (most recent)
        """
        return self.current_frame
    
    def register_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Register a callback function to be called for each frame
        """
        self.frame_callbacks.append(callback)
    
    def unregister_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Unregister a frame callback
        """
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def get_camera_info(self) -> Dict:
        """
        Get camera information
        """
        if not self.cap:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'is_running': self.is_running,
            'source': self.camera_source
        }
    
    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """
        Save a frame to file
        """
        try:
            cv2.imwrite(filename, frame)
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False
    
    def encode_frame_to_jpeg(self, frame: np.ndarray, quality: int = 80) -> Optional[bytes]:
        """
        Encode frame to JPEG bytes for streaming
        """
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)
            
            if result:
                return encoded_img.tobytes()
            else:
                return None
                
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    async def get_frame_async(self) -> Optional[Dict]:
        """
        Async version of get_latest_frame
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_latest_frame)
    
    def __enter__(self):
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()
