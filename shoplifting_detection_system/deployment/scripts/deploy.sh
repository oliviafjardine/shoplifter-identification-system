#!/bin/bash
set -e

echo "🚀 Deploying Shoplifting Detection System..."

# Build and deploy
docker-compose -f deployment/docker/docker-compose.prod.yml down
docker-compose -f deployment/docker/docker-compose.prod.yml build
docker-compose -f deployment/docker/docker-compose.prod.yml up -d

echo "✅ Deployment completed!"
echo "🌐 Application available at: http://localhost"
echo "📊 Health check: http://localhost/health"