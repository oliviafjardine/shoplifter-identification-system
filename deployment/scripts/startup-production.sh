#!/bin/bash

# Production Startup Script for Shoplifting Detection System
# This script handles the startup process for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Set default values
MODE=${1:-production}
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8080}
WORKERS=${WORKERS:-4}
LOG_LEVEL=${LOG_LEVEL:-info}

log "🚀 Starting Shoplifting Detection System in $MODE mode"

# Activate virtual environment
source /home/shoplifter/venv/bin/activate

# Change to application directory
cd /home/shoplifter/app

# Check if required files exist
if [ ! -f "main.py" ]; then
    error "main.py not found in application directory"
    exit 1
fi

# Create necessary directories
mkdir -p logs evidence models data/models config static uploads backups

# Set permissions
chmod 755 logs evidence models data config static uploads backups

# Wait for database if DATABASE_URL is set
if [ ! -z "$DATABASE_URL" ]; then
    log "⏳ Waiting for database connection..."
    
    # Extract database host and port from DATABASE_URL
    DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    
    if [ ! -z "$DB_HOST" ] && [ ! -z "$DB_PORT" ]; then
        timeout=60
        while ! nc -z $DB_HOST $DB_PORT; do
            timeout=$((timeout - 1))
            if [ $timeout -le 0 ]; then
                error "Database connection timeout"
                exit 1
            fi
            sleep 1
        done
        log "✅ Database connection established"
    fi
fi

# Wait for Redis if REDIS_URL is set
if [ ! -z "$REDIS_URL" ]; then
    log "⏳ Waiting for Redis connection..."
    
    # Extract Redis host and port from REDIS_URL
    REDIS_HOST=$(echo $REDIS_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    REDIS_PORT=$(echo $REDIS_URL | sed -n 's/.*:\([0-9]*\)$/\1/p')
    
    if [ ! -z "$REDIS_HOST" ] && [ ! -z "$REDIS_PORT" ]; then
        timeout=30
        while ! nc -z $REDIS_HOST $REDIS_PORT; do
            timeout=$((timeout - 1))
            if [ $timeout -le 0 ]; then
                error "Redis connection timeout"
                exit 1
            fi
            sleep 1
        done
        log "✅ Redis connection established"
    fi
fi

# Check if trained model exists
if [ ! -f "data/models/shoplifting_model.pkl" ]; then
    warn "⚠️  Trained model not found. Training a basic model..."
    python train_model.py --quick-train
    if [ $? -eq 0 ]; then
        log "✅ Basic model trained successfully"
    else
        error "Failed to train basic model"
        exit 1
    fi
fi

# Validate configuration
log "🔧 Validating configuration..."

# Check camera access (if not in container)
if [ -z "$CONTAINER" ]; then
    if [ ! -e "/dev/video0" ]; then
        warn "⚠️  No camera device found at /dev/video0"
        warn "⚠️  System will run in demo mode"
        export DEMO_MODE=true
    fi
fi

# Start the application based on mode
case $MODE in
    "production")
        log "🏭 Starting production server with Gunicorn"
        exec gunicorn main:app \
            --bind $HOST:$PORT \
            --workers $WORKERS \
            --worker-class uvicorn.workers.UvicornWorker \
            --access-logfile logs/access.log \
            --error-logfile logs/error.log \
            --log-level $LOG_LEVEL \
            --timeout 120 \
            --keep-alive 2 \
            --max-requests 1000 \
            --max-requests-jitter 50 \
            --preload
        ;;
    "development")
        log "🔧 Starting development server with Uvicorn"
        exec uvicorn main:app \
            --host $HOST \
            --port $PORT \
            --reload \
            --log-level $LOG_LEVEL
        ;;
    "test")
        log "🧪 Running tests"
        exec python -m pytest tests/ -v
        ;;
    *)
        error "Unknown mode: $MODE"
        error "Available modes: production, development, test"
        exit 1
        ;;
esac
