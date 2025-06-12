#!/bin/bash
set -e

echo "🔍 Running health checks..."

# Check if containers are running
if ! docker-compose -f deployment/docker/docker-compose.prod.yml ps | grep -q "Up"; then
    echo "❌ Containers are not running"
    exit 1
fi

# Check application health
if ! curl -f http://localhost/health > /dev/null 2>&1; then
    echo "❌ Application health check failed"
    exit 1
fi

echo "✅ All health checks passed!"