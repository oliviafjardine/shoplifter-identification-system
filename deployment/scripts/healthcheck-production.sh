#!/bin/bash

# Production Health Check Script for Shoplifting Detection System
# This script performs comprehensive health checks for the application

set -e

# Configuration
HOST=${API_HOST:-localhost}
PORT=${API_PORT:-8080}
TIMEOUT=10
MAX_RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Health check functions
check_http_endpoint() {
    local endpoint=$1
    local expected_status=${2:-200}
    
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time $TIMEOUT \
        "http://$HOST:$PORT$endpoint" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        return 0
    else
        return 1
    fi
}

check_api_health() {
    if check_http_endpoint "/health" 200; then
        log_success "API health endpoint responding"
        return 0
    else
        log_error "API health endpoint not responding"
        return 1
    fi
}

check_api_ready() {
    if check_http_endpoint "/ready" 200; then
        log_success "API ready endpoint responding"
        return 0
    else
        log_error "API ready endpoint not responding"
        return 1
    fi
}

check_dashboard() {
    if check_http_endpoint "/" 200; then
        log_success "Dashboard endpoint responding"
        return 0
    else
        log_error "Dashboard endpoint not responding"
        return 1
    fi
}

check_api_docs() {
    if check_http_endpoint "/docs" 200; then
        log_success "API documentation endpoint responding"
        return 0
    else
        log_warning "API documentation endpoint not responding"
        return 1
    fi
}

check_metrics() {
    if check_http_endpoint "/metrics" 200; then
        log_success "Metrics endpoint responding"
        return 0
    else
        log_warning "Metrics endpoint not responding"
        return 1
    fi
}

check_process() {
    if pgrep -f "main:app" > /dev/null; then
        log_success "Application process running"
        return 0
    else
        log_error "Application process not found"
        return 1
    fi
}

check_memory_usage() {
    local memory_usage=$(ps -o pid,ppid,cmd,%mem --sort=-%mem | grep -E "(python|gunicorn|uvicorn)" | head -1 | awk '{print $4}')
    
    if [ ! -z "$memory_usage" ]; then
        local memory_threshold=80
        if (( $(echo "$memory_usage > $memory_threshold" | bc -l) )); then
            log_warning "High memory usage: ${memory_usage}%"
            return 1
        else
            log_success "Memory usage normal: ${memory_usage}%"
            return 0
        fi
    else
        log_warning "Could not determine memory usage"
        return 1
    fi
}

check_disk_space() {
    local disk_usage=$(df /home/shoplifter/app | tail -1 | awk '{print $5}' | sed 's/%//')
    local disk_threshold=90
    
    if [ "$disk_usage" -gt "$disk_threshold" ]; then
        log_warning "High disk usage: ${disk_usage}%"
        return 1
    else
        log_success "Disk usage normal: ${disk_usage}%"
        return 0
    fi
}

check_log_files() {
    local log_dir="/home/shoplifter/app/logs"
    
    if [ -d "$log_dir" ]; then
        local log_files=$(find "$log_dir" -name "*.log" -mtime -1 | wc -l)
        if [ "$log_files" -gt 0 ]; then
            log_success "Log files are being written"
            return 0
        else
            log_warning "No recent log files found"
            return 1
        fi
    else
        log_warning "Log directory not found"
        return 1
    fi
}

# Main health check function
main() {
    local exit_code=0
    local checks_passed=0
    local total_checks=0
    
    echo "🏥 Starting health check for Shoplifting Detection System"
    echo "=================================================="
    
    # Critical checks (must pass)
    echo "🔍 Running critical checks..."
    
    total_checks=$((total_checks + 1))
    if check_process; then
        checks_passed=$((checks_passed + 1))
    else
        exit_code=1
    fi
    
    total_checks=$((total_checks + 1))
    if check_api_health; then
        checks_passed=$((checks_passed + 1))
    else
        exit_code=1
    fi
    
    total_checks=$((total_checks + 1))
    if check_api_ready; then
        checks_passed=$((checks_passed + 1))
    else
        exit_code=1
    fi
    
    total_checks=$((total_checks + 1))
    if check_dashboard; then
        checks_passed=$((checks_passed + 1))
    else
        exit_code=1
    fi
    
    # Optional checks (warnings only)
    echo ""
    echo "🔍 Running optional checks..."
    
    total_checks=$((total_checks + 1))
    if check_api_docs; then
        checks_passed=$((checks_passed + 1))
    fi
    
    total_checks=$((total_checks + 1))
    if check_metrics; then
        checks_passed=$((checks_passed + 1))
    fi
    
    total_checks=$((total_checks + 1))
    if check_memory_usage; then
        checks_passed=$((checks_passed + 1))
    fi
    
    total_checks=$((total_checks + 1))
    if check_disk_space; then
        checks_passed=$((checks_passed + 1))
    fi
    
    total_checks=$((total_checks + 1))
    if check_log_files; then
        checks_passed=$((checks_passed + 1))
    fi
    
    # Summary
    echo ""
    echo "=================================================="
    echo "Health check summary: $checks_passed/$total_checks checks passed"
    
    if [ $exit_code -eq 0 ]; then
        log_success "All critical health checks passed"
    else
        log_error "Some critical health checks failed"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"
