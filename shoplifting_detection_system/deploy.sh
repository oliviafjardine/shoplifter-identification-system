#!/bin/bash

# Production Deployment Script for Shoplifting Detection System
# This script handles the complete deployment process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="shoplifting-detection-system"
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE=".env.production"

# Logging functions
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

header() {
    echo -e "${PURPLE}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    header "🔍 Checking Prerequisites"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log "✅ Docker is installed"
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    log "✅ Docker Compose is installed"
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    log "✅ Docker daemon is running"
    
    # Check available disk space (minimum 5GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=5242880  # 5GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        error "Insufficient disk space. At least 5GB required."
        exit 1
    fi
    log "✅ Sufficient disk space available"
}

# Function to setup environment
setup_environment() {
    header "🔧 Setting Up Environment"
    
    cd "$SCRIPT_DIR"
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        warn "Environment file $ENV_FILE not found. Creating from template..."
        cp "$ENV_FILE" .env
        warn "⚠️  Please edit .env file with your production settings before continuing!"
        warn "⚠️  Pay special attention to passwords and secret keys!"
        read -p "Press Enter after you've configured the .env file..."
    else
        log "✅ Environment file found"
    fi
    
    # Create necessary directories
    mkdir -p logs evidence models config static uploads backups
    mkdir -p deployment/nginx/ssl
    mkdir -p deployment/prometheus
    mkdir -p deployment/grafana/dashboards
    mkdir -p deployment/grafana/datasources
    
    log "✅ Directories created"
    
    # Set proper permissions
    chmod 755 logs evidence models config static uploads backups
    chmod +x deployment/scripts/*.sh 2>/dev/null || true
    
    log "✅ Permissions set"
}

# Function to generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    header "🔐 Generating SSL Certificates"
    
    SSL_DIR="deployment/nginx/ssl"
    
    if [ ! -f "$SSL_DIR/cert.pem" ] || [ ! -f "$SSL_DIR/key.pem" ]; then
        info "Generating self-signed SSL certificates..."
        
        openssl req -x509 -newkey rsa:4096 -keyout "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log "✅ SSL certificates generated"
        warn "⚠️  Using self-signed certificates. Replace with proper certificates for production!"
    else
        log "✅ SSL certificates already exist"
    fi
}

# Function to build Docker images
build_images() {
    header "🏗️  Building Docker Images"
    
    info "Building production image..."
    docker build -f Dockerfile.production -t "$PROJECT_NAME:latest" .
    
    log "✅ Docker images built successfully"
}

# Function to start services
start_services() {
    header "🚀 Starting Services"
    
    # Start core services first
    info "Starting core services (database, cache)..."
    docker-compose -f "$COMPOSE_FILE" up -d postgres redis
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 10
    
    # Start main application
    info "Starting main application..."
    docker-compose -f "$COMPOSE_FILE" up -d shoplifting-app nginx
    
    # Start monitoring (if enabled)
    if [ "$1" = "--with-monitoring" ]; then
        info "Starting monitoring services..."
        docker-compose -f "$COMPOSE_FILE" --profile monitoring up -d
    fi
    
    log "✅ All services started"
}

# Function to check service health
check_health() {
    header "🏥 Checking Service Health"
    
    info "Waiting for services to be healthy..."
    
    # Wait for application to be ready
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            log "✅ Application is healthy"
            break
        fi
        timeout=$((timeout - 1))
        sleep 1
    done
    
    if [ $timeout -eq 0 ]; then
        error "Application health check failed"
        return 1
    fi
    
    # Check other services
    docker-compose -f "$COMPOSE_FILE" ps
}

# Function to show deployment information
show_deployment_info() {
    header "📋 Deployment Information"
    
    echo "🌐 Application URLs:"
    echo "   📊 Main Dashboard: http://localhost:8080"
    echo "   📚 API Documentation: http://localhost:8080/docs"
    echo "   📈 Health Check: http://localhost:8080/health"
    echo ""
    
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q prometheus; then
        echo "📊 Monitoring URLs:"
        echo "   📈 Prometheus: http://localhost:9091"
        echo "   📊 Grafana: http://localhost:3000 (admin/admin123)"
        echo ""
    fi
    
    echo "🔧 Management Commands:"
    echo "   📋 View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "   🔄 Restart: docker-compose -f $COMPOSE_FILE restart"
    echo "   🛑 Stop: docker-compose -f $COMPOSE_FILE down"
    echo "   🗑️  Clean up: docker-compose -f $COMPOSE_FILE down -v"
    echo ""
    
    log "✅ Deployment completed successfully!"
}

# Function to stop services
stop_services() {
    header "🛑 Stopping Services"
    
    docker-compose -f "$COMPOSE_FILE" down
    
    log "✅ All services stopped"
}

# Function to clean up
cleanup() {
    header "🗑️  Cleaning Up"
    
    read -p "This will remove all containers, volumes, and data. Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
        docker system prune -f
        log "✅ Cleanup completed"
    else
        info "Cleanup cancelled"
    fi
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            setup_environment
            generate_ssl_certificates
            build_images
            start_services "$2"
            sleep 5
            check_health
            show_deployment_info
            ;;
        "start")
            start_services "$2"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            start_services "$2"
            ;;
        "health")
            check_health
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" logs -f "${2:-shoplifting-app}"
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  deploy              Full deployment (default)"
            echo "  start               Start services"
            echo "  stop                Stop services"
            echo "  restart             Restart services"
            echo "  health              Check service health"
            echo "  logs [service]      View logs"
            echo "  cleanup             Remove all containers and data"
            echo "  help                Show this help"
            echo ""
            echo "Options:"
            echo "  --with-monitoring   Include monitoring services"
            ;;
        *)
            error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
