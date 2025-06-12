#!/bin/bash

# Quick Start Script for Shoplifting Detection System
# This script provides a fast way to get the system running for development/testing

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

header() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker daemon is not running. Please start Docker first."
        exit 1
    fi
}

# Quick setup for development
quick_setup() {
    header "🚀 Quick Start - Shoplifting Detection System"
    
    log "Checking prerequisites..."
    check_docker
    
    log "Creating necessary directories..."
    mkdir -p logs evidence models data/models config static uploads
    
    log "Setting up environment..."
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Quick Start Configuration
DEBUG=true
LOG_LEVEL=debug
DEMO_MODE=true
WORKERS=2
API_HOST=0.0.0.0
API_PORT=8080

# Database (using SQLite for quick start)
DATABASE_URL=sqlite:///./data/shoplifting.db

# Redis (optional for quick start)
REDIS_URL=redis://localhost:6379

# Security (development only)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production

# Features
ENABLE_FACE_DETECTION=true
ENABLE_BODY_DETECTION=true
ENABLE_MOTION_DETECTION=false
ENABLE_EDGE_DETECTION=false
ENABLE_SHOPLIFTING_DETECTION=true
ENABLE_NOTIFICATIONS=false
ENABLE_EVIDENCE_STORAGE=true
EOF
        log "✅ Environment file created"
    else
        log "✅ Environment file already exists"
    fi
    
    log "Installing Python dependencies..."
    if command -v python3 &> /dev/null; then
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            log "✅ Virtual environment created"
        fi
        
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        log "✅ Dependencies installed"
    else
        warn "⚠️  Python3 not found. Using Docker instead..."
        USE_DOCKER=true
    fi
}

# Start with Python (development)
start_python() {
    header "🐍 Starting with Python"
    
    source venv/bin/activate
    
    log "Training basic model..."
    python train_model.py --quick-train
    
    log "Starting application..."
    info "🌐 Application will be available at: http://localhost:8080"
    info "📚 API documentation at: http://localhost:8080/docs"
    info "🛑 Press Ctrl+C to stop"
    
    python main.py
}

# Start with Docker (production-like)
start_docker() {
    header "🐳 Starting with Docker"
    
    log "Building Docker image..."
    docker build -f Dockerfile.production -t shoplifting-detection:latest .
    
    log "Starting application..."
    docker run -it --rm \
        -p 8080:8080 \
        -v $(pwd)/logs:/home/shoplifter/app/logs \
        -v $(pwd)/evidence:/home/shoplifter/app/evidence \
        -v $(pwd)/models:/home/shoplifter/app/models \
        -v $(pwd)/data:/home/shoplifter/app/data \
        --env-file .env \
        shoplifting-detection:latest
}

# Main menu
show_menu() {
    header "🎯 Quick Start Options"
    
    echo "Choose how you want to run the system:"
    echo ""
    echo "1) 🐍 Python (Development) - Fast startup, good for development"
    echo "2) 🐳 Docker (Production-like) - Containerized, closer to production"
    echo "3) 🚀 Full Production Deployment - Complete setup with all services"
    echo "4) ❓ Help - Show detailed information"
    echo "5) 🚪 Exit"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            if [ "$USE_DOCKER" = "true" ]; then
                warn "⚠️  Python3 not available, falling back to Docker"
                start_docker
            else
                start_python
            fi
            ;;
        2)
            start_docker
            ;;
        3)
            info "Running full production deployment..."
            chmod +x deploy.sh
            ./deploy.sh deploy
            ;;
        4)
            show_help
            ;;
        5)
            log "👋 Goodbye!"
            exit 0
            ;;
        *)
            warn "⚠️  Invalid choice. Please try again."
            show_menu
            ;;
    esac
}

# Show help information
show_help() {
    header "📖 Help Information"
    
    echo "🎯 Quick Start Options:"
    echo ""
    echo "1. Python Development:"
    echo "   - Fastest startup time"
    echo "   - Good for development and testing"
    echo "   - Requires Python 3.10+"
    echo "   - Uses SQLite database"
    echo "   - Demo mode enabled (no camera required)"
    echo ""
    echo "2. Docker Development:"
    echo "   - Containerized environment"
    echo "   - Closer to production setup"
    echo "   - No local Python installation required"
    echo "   - Uses SQLite database"
    echo "   - Demo mode enabled"
    echo ""
    echo "3. Full Production:"
    echo "   - Complete production setup"
    echo "   - PostgreSQL database"
    echo "   - Redis caching"
    echo "   - Nginx reverse proxy"
    echo "   - SSL certificates"
    echo "   - Optional monitoring"
    echo ""
    echo "🔧 System Requirements:"
    echo "   - Docker (for options 2 & 3)"
    echo "   - Python 3.10+ (for option 1)"
    echo "   - 4GB+ RAM recommended"
    echo "   - Camera (optional, demo mode available)"
    echo ""
    echo "🌐 Access Points:"
    echo "   - Main Dashboard: http://localhost:8080"
    echo "   - API Docs: http://localhost:8080/docs"
    echo "   - Health Check: http://localhost:8080/health"
    echo ""
    
    read -p "Press Enter to return to menu..."
    show_menu
}

# Main execution
main() {
    # Check if running with arguments
    case "${1:-menu}" in
        "python"|"py")
            quick_setup
            start_python
            ;;
        "docker")
            quick_setup
            start_docker
            ;;
        "production"|"prod")
            chmod +x deploy.sh
            ./deploy.sh deploy
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            quick_setup
            show_menu
            ;;
    esac
}

# Run main function
main "$@"
