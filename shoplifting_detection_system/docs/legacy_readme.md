# Shoplifting Detection System

A comprehensive AI-powered shoplifting detection system that uses computer vision, behavior analysis, and anomaly detection to identify suspicious activities in real-time.

## 🚀 Features

### Core Detection Capabilities
- **Object Detection**: YOLO-based detection of people and items
- **Person Tracking**: Multi-object tracking across video frames
- **Behavior Analysis**: Detection of suspicious behaviors including:
  - Crouching/hiding behavior
  - Loitering in specific areas
  - Erratic movement patterns
  - Suspicious hand movements
  - Item proximity analysis
- **Anomaly Detection**: Machine learning-based anomaly detection using Isolation Forest
- **Real-time Alerts**: Instant notifications for suspicious activities

### Technical Features
- **FastAPI Backend**: High-performance async API server
- **WebSocket Support**: Real-time communication for live alerts
- **PostgreSQL Database**: Persistent storage for events, alerts, and tracking data
- **OpenCV Integration**: Advanced computer vision processing
- **Responsive Web Dashboard**: Modern, mobile-friendly interface

### Dashboard Features
- Live video feed with detection overlays
- Real-time alert notifications
- Alert management and acknowledgment
- System statistics and monitoring
- Historical data analysis

## �️ Tech Stack & Tools

### Backend Technologies
- **Python 3.8+** - Core programming language
- **FastAPI** - Modern, high-performance web framework for APIs
- **Uvicorn** - ASGI server for FastAPI applications
- **SQLAlchemy** - Python SQL toolkit and Object-Relational Mapping (ORM)
- **Pydantic** - Data validation using Python type annotations
- **Asyncio** - Asynchronous programming support

### Computer Vision & AI
- **OpenCV (cv2)** - Computer vision library for image/video processing
- **Ultralytics YOLO** - State-of-the-art object detection model
- **NumPy** - Numerical computing library for array operations
- **Scikit-learn** - Machine learning library for anomaly detection
- **Isolation Forest** - Unsupervised anomaly detection algorithm
- **PyTorch** - Deep learning framework (YOLO backend)

### Database & Storage
- **PostgreSQL** - Primary relational database
- **SQLite** - Alternative lightweight database option
- **Alembic** - Database migration tool for SQLAlchemy

### Frontend Technologies
- **HTML** - Modern web markup
- **CSS** - Styling with Flexbox/Grid layouts
- **JavaScript** - Client-side programming
- **WebSocket API** - Real-time bidirectional communication
- **Fetch API** - Modern HTTP client for API calls
- **Responsive Design** - Mobile-friendly interface

### Development & Deployment Tools
- **Docker** - Containerization platform
- **Docker Compose** - Multi-container application orchestration
- **Git** - Version control system
- **pip** - Python package manager
- **Virtual Environment (venv)** - Python environment isolation

### System Integration
- **WebSocket** - Real-time communication protocol
- **REST API** - HTTP-based API architecture
- **JSON** - Data interchange format
- **Environment Variables** - Configuration management
- **Logging** - Application monitoring and debugging

### Hardware Requirements
- **Camera/Webcam** - Video input source
- **CPU/GPU** - Processing power for AI inference
- **RAM** - Memory for video processing and ML models
- **Storage** - Database and model file storage

### Optional Enhancements
- **CUDA** - GPU acceleration for YOLO inference
- **Redis** - Caching and session management
- **Nginx** - Reverse proxy and load balancing
- **SSL/TLS** - Secure communication
- **Cloud Storage** - Scalable file storage solutions

### Development Workflow
- **Python Virtual Environment** - Isolated development environment
- **Requirements.txt** - Dependency management
- **Configuration Files** - Environment-specific settings
- **Automated Setup Scripts** - Easy installation process
- **Testing Framework** - Unit and integration tests

## �📋 Requirements

- Python 3.8+
- PostgreSQL (or Docker for easy setup)
- Webcam or IP camera
- Modern web browser

## 🛠️ Installation

### Quick Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd shoplifter-identifier
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```

3. **Activate virtual environment**
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Start the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8000`

### Manual Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup PostgreSQL**
   ```bash
   # Using Docker (recommended)
   docker-compose up -d postgres
   
   # Or install PostgreSQL manually and create database
   ```

4. **Configure environment**
   ```bash
   # Edit .env file with your settings
   cp .env.example .env
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

Edit the `.env` file to customize the system:

```env
# Database Configuration
DATABASE_URL=postgresql://shoplifter_user:shoplifter_pass@localhost:5432/shoplifter_db

# Camera Configuration
CAMERA_SOURCE=0  # 0 for webcam, or path to video file

# Detection Thresholds
ALERT_THRESHOLD=0.7  # Suspicion threshold (0.0-1.0)

# Model Configuration
MODEL_PATH=yolov8n.pt  # YOLO model path

# Debug Mode
DEBUG=True
```

### Advanced Configuration

You can also modify `config.py` for more detailed settings:

- **Detection thresholds**: Adjust sensitivity for different behaviors
- **Tracking parameters**: Fine-tune person tracking
- **Alert settings**: Configure alert cooldown periods
- **Database settings**: Customize database connection

## 🎯 Usage

### Starting the System

1. **Ensure camera is connected** and not being used by other applications
2. **Start the database** (if using Docker): `docker-compose up -d postgres`
3. **Run the application**: `python main.py`
4. **Access the dashboard**: Open `http://localhost:8000` in your browser

### Using the Dashboard

#### Live Video Feed
- View real-time camera feed with detection overlays
- See detected people and objects highlighted with bounding boxes
- Monitor detection confidence scores

#### Alert Management
- Receive real-time alerts for suspicious activities
- View detailed alert information including:
  - Timestamp and severity level
  - Detected behaviors and confidence scores
  - Person tracking information
- Acknowledge alerts to mark them as reviewed
- Filter and search through alert history

#### System Statistics
- Monitor system performance metrics
- View alert statistics by severity level
- Track acknowledgment rates
- Monitor camera and tracking status

### API Endpoints

The system provides a REST API for integration:

- `GET /api/alerts` - Retrieve recent alerts
- `POST /api/alerts/{id}/acknowledge` - Acknowledge an alert
- `GET /api/stats` - Get system statistics
- `GET /api/camera/info` - Get camera information
- `GET /video_feed` - Live video stream
- `WebSocket /ws` - Real-time updates

## 🧠 How It Works

### Detection Pipeline

1. **Video Capture**: Captures frames from camera/video source
2. **Object Detection**: Uses YOLO to detect people and objects
3. **Person Tracking**: Tracks individuals across frames
4. **Behavior Analysis**: Analyzes movement patterns and behaviors
5. **Anomaly Detection**: Uses ML to identify unusual patterns
6. **Alert Generation**: Creates alerts for suspicious activities
7. **Real-time Notification**: Sends alerts to dashboard via WebSocket

### Behavior Detection

The system detects various suspicious behaviors:

- **Crouching**: Detected by analyzing bounding box aspect ratios
- **Loitering**: Identified by tracking time spent in small areas
- **Erratic Movement**: Detected through movement pattern analysis
- **Hand Movements**: Simplified detection based on bounding box changes
- **Item Proximity**: Analysis of person-to-object distances

### Machine Learning

- **Anomaly Detection**: Uses Isolation Forest algorithm
- **Feature Engineering**: Extracts movement, temporal, and behavioral features
- **Adaptive Learning**: Model updates with new data over time
- **Confidence Scoring**: Provides probability scores for detections

## 🔧 Troubleshooting

### Common Issues

**Camera not working**
- Try different `CAMERA_SOURCE` values (0, 1, 2, etc.)
- Ensure camera isn't being used by other applications
- Check camera permissions

**Database connection errors**
- Verify PostgreSQL is running: `docker-compose ps`
- Check database credentials in `.env`
- Ensure database exists and is accessible

**High CPU usage**
- Reduce video resolution in camera settings
- Adjust processing frame rate
- Use GPU acceleration if available

**False alerts**
- Adjust `ALERT_THRESHOLD` in `.env`
- Fine-tune detection thresholds in `config.py`
- Allow system to learn normal patterns over time

### Performance Optimization

- Use GPU acceleration for YOLO inference
- Optimize video resolution and frame rate
- Implement frame skipping for better performance
- Use hardware-accelerated video decoding

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feed   │───▶│   FastAPI App    │───▶│   Web Dashboard │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Detection       │
                    │  Pipeline        │
                    │  ┌─────────────┐ │
                    │  │ YOLO        │ │
                    │  │ Detection   │ │
                    │  └─────────────┘ │
                    │  ┌─────────────┐ │
                    │  │ Person      │ │
                    │  │ Tracking    │ │
                    │  └─────────────┘ │
                    │  ┌─────────────┐ │
                    │  │ Behavior    │ │
                    │  │ Analysis    │ │
                    │  └─────────────┘ │
                    │  ┌─────────────┐ │
                    │  │ Anomaly     │ │
                    │  │ Detection   │ │
                    │  └─────────────┘ │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   PostgreSQL     │
                    │   Database       │
                    └──────────────────┘
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review system logs for error messages
- Open an issue on GitHub
- Check camera and system requirements

## 🔮 Future Enhancements

- **Advanced Pose Estimation**: More accurate hand movement detection
- **Facial Recognition**: Identity tracking and blacklist management
- **Zone-based Detection**: Configurable detection zones
- **Mobile App**: Mobile dashboard and notifications
- **Cloud Integration**: Cloud storage and analytics
- **Advanced ML Models**: Custom trained models for specific scenarios
