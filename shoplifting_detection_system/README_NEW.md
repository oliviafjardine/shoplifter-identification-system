# 🛡️ Professional Shoplifting Detection System

A consolidated AI-powered shoplifting detection system with live camera integration and professional surveillance dashboard.

## ✨ Features

- **🎯 Live Camera Detection**: Real-time analysis using trained ML model
- **📹 Professional Dashboard**: Modern surveillance interface matching industry standards
- **🤖 AI-Powered Analysis**: Trained on Kaggle shoplifting dataset
- **🚨 Real-time Alerts**: Instant notifications for suspicious behavior
- **📊 Advanced Analytics**: Comprehensive statistics and reporting
- **🎨 Modern UI**: Dark theme professional surveillance interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Camera (webcam or USB camera)
- Required packages (see requirements.txt)

### Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Train the Model** (First time only):
```bash
python train_model.py
```
This will download the Kaggle dataset and train the shoplifting detection model.

3. **Run the System**:
```bash
python main.py
```

4. **Access Dashboard**:
   - 🌐 **Main Dashboard**: http://localhost:8080
   - 📚 **API Documentation**: http://localhost:8080/docs
   - 📊 **System Statistics**: http://localhost:8080/api/stats

## 🎯 System Overview

### Core Components

- **`main.py`**: Consolidated system with live camera detection and professional dashboard
- **`train_model.py`**: Dataset-only training script using Kaggle shoplifting videos
- **`ml/models/`**: Trained model storage
- **Professional UI**: Modern surveillance dashboard with real-time updates

### Key Features

#### 🔴 Live Camera Detection
- Real-time people detection using OpenCV
- AI behavior analysis with trained ML model
- Suspicious activity scoring and alerts
- Camera connection status monitoring

#### 📱 Professional Dashboard
- **Multi-camera grid layout** (main + 3 secondary feeds)
- **Real-time alerts panel** with scrolling and severity levels
- **Live statistics** (people count, accuracy, uptime)
- **Professional controls** (detection toggle, export, API access)
- **Modern dark theme** matching surveillance industry standards

#### 🤖 AI Model
- Trained exclusively on Kaggle shoplifting dataset
- Random Forest classifier with behavior feature extraction
- Real-time prediction with confidence scoring
- Automatic alert generation for suspicious behavior

## 📊 Dashboard Interface

The professional dashboard includes:

- **Left Sidebar**: Navigation (Live View, Playback, Events, Reports, Config)
- **Top Header**: System title, search, live monitoring indicator
- **Camera Grid**: Main camera (2x2 space) + 3 secondary cameras
- **Right Sidebar**: 
  - 🚨 Alerts with severity levels and scrolling
  - 📊 Real-time statistics
  - 🎛️ System controls

## 🔧 API Endpoints

- `GET /`: Professional dashboard interface
- `GET /api/dashboard_data`: Real-time surveillance data
- `GET /api/stats`: Comprehensive system statistics
- `POST /api/clear-alerts`: Clear all alerts

## 🎯 Training Process

The system uses only the Kaggle dataset for training:

1. **Dataset**: `minhajuddinmeraj/anomalydetectiondatasetucf`
2. **Focus**: `Anomaly-Videos-Part-4/Anomaly-Videos-Part-4Shoplifting` folder
3. **Features**: Motion analysis, edge detection, behavioral patterns
4. **Model**: Random Forest with balanced classes
5. **Output**: Trained model saved to `ml/models/continuous_model.pkl`

## 🚨 Alert System

- **Critical**: Suspicious behavior detected by AI (red)
- **Warning**: Unusual patterns or restricted access (orange)
- **Info**: General notifications and status updates (blue)
- **Real-time**: Automatic scrolling with timestamps and camera IDs

## 🎨 UI Design

Professional surveillance aesthetic:
- **Dark theme**: #1a1a1a background, #2d2d2d panels
- **Modern typography**: System fonts for professional appearance
- **Color coding**: Green for active, red for alerts, blue for info
- **Responsive**: Adapts to different screen sizes
- **No animations**: Clean, distraction-free interface

## 📈 System Status

- **Camera Connection**: Live status with error handling
- **Model Accuracy**: Real-time display of AI performance
- **People Detection**: Live count across all cameras
- **Alert Management**: Automatic cleanup and organization

## 🛠️ Troubleshooting

- **No Camera**: System runs in demo mode with simulated data
- **Model Missing**: Run `train_model.py` to train from dataset
- **Port Conflict**: System uses port 8080 by default

## 📝 File Structure

```
shoplifting_detection_system/
├── main.py              # Main consolidated system
├── train_model.py       # Dataset training script
├── ml/models/           # Trained model storage
├── requirements.txt     # Dependencies
└── README_NEW.md       # This file
```

## 🎉 Success Metrics

- ✅ **Professional UI**: Matches surveillance industry standards
- ✅ **Live Detection**: Real camera integration with AI analysis
- ✅ **Dataset Training**: Uses only Kaggle shoplifting videos
- ✅ **Single Port**: Consolidated system on port 8080
- ✅ **Clean Structure**: Focused on essential files only
- ✅ **Scrolling Alerts**: Fixed UI with proper alert management

## 🔄 Current Status

The system is now fully operational with:
- ✅ Professional dashboard running on port 8080
- ✅ Live camera detection with trained ML model
- ✅ Real-time alerts with proper scrolling
- ✅ Clean file structure with unnecessary files removed
- ✅ Dataset-only training process
- ✅ Modern surveillance interface design
