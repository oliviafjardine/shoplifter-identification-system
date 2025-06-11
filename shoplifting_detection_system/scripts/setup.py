#!/usr/bin/env python3
"""
Setup script for Shoplifting Detection System
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(
        f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")

    if not venv_path.exists():
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            return False

    # Activation command varies by OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"

    print(f"✅ Virtual environment ready")
    print(f"To activate: {activate_cmd}")

    return pip_cmd


def install_dependencies(pip_cmd):
    """Install Python dependencies"""
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python dependencies")


def setup_database():
    """Setup PostgreSQL database using Docker"""
    print("\n🐳 Setting up PostgreSQL database...")

    # Check if Docker is available
    if not run_command("docker --version", "Checking Docker installation"):
        print("⚠️  Docker not found. Please install Docker to use the database.")
        print("Alternatively, you can set up PostgreSQL manually and update the DATABASE_URL in .env")
        return False

    # Start PostgreSQL container
    if not run_command("docker-compose up -d postgres", "Starting PostgreSQL container"):
        return False

    print("✅ PostgreSQL database is running")
    print("Database URL: postgresql://shoplifter_user:shoplifter_pass@localhost:5432/shoplifter_db")
    return True


def download_yolo_model():
    """Download YOLO model if not present"""
    model_path = Path("yolov8n.pt")
    if not model_path.exists():
        print("\n📥 Downloading YOLO model...")
        try:
            from ultralytics import YOLO
            # This will automatically download the model
            model = YOLO('yolov8n.pt')
            print("✅ YOLO model downloaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to download YOLO model: {e}")
            return False
    else:
        print("✅ YOLO model already exists")
        return True


def create_directories():
    """Create necessary directories"""
    directories = ['static/images', 'logs', 'models']

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ Created necessary directories")
    return True


def main():
    """Main setup function"""
    print("🚀 Setting up Shoplifting Detection System")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Setup virtual environment
    pip_cmd = setup_virtual_environment()
    if not pip_cmd:
        sys.exit(1)

    # Install dependencies
    if not install_dependencies(pip_cmd):
        sys.exit(1)

    # Create directories
    if not create_directories():
        sys.exit(1)

    # Setup database
    database_setup = setup_database()

    # Download YOLO model (after installing dependencies)
    model_setup = download_yolo_model()

    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")

    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")

    if database_setup:
        print("2. Database is ready (PostgreSQL running in Docker)")
    else:
        print("2. Set up PostgreSQL manually and update DATABASE_URL in .env")

    if model_setup:
        print("3. YOLO model is ready")
    else:
        print("3. Download YOLO model manually or check internet connection")

    print("4. Start the application:")
    print("   python main.py")
    print("\n5. Open your browser and go to: http://localhost:8000")

    print("\n📝 Configuration:")
    print("- Edit .env file to customize settings")
    print("- Camera source: Set CAMERA_SOURCE=0 for webcam or provide video file path")
    print("- Alert threshold: Adjust ALERT_THRESHOLD (0.0-1.0)")

    print("\n🔧 Troubleshooting:")
    print("- If camera doesn't work, try different CAMERA_SOURCE values (0, 1, 2, etc.)")
    print("- Check logs for detailed error messages")
    print("- Ensure your camera is not being used by other applications")


if __name__ == "__main__":
    main()
