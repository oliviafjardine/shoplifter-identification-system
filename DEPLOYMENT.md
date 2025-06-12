# 🚀 Production Deployment Guide

## Shoplifting Detection System - Production Deployment

This guide provides comprehensive instructions for deploying the Shoplifting Detection System in a production environment.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 20GB free space
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for better performance
- **Camera**: USB camera or IP camera (optional for demo mode)

### Software Requirements

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Git**: For cloning the repository
- **OpenSSL**: For SSL certificate generation

### Network Requirements

- **Ports**: 80 (HTTP), 443 (HTTPS), 8080 (Application)
- **Internet**: Required for downloading Docker images and dependencies

## 🛠️ Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd shoplifting_detection_system
```

### 2. Make Deployment Script Executable

```bash
chmod +x deploy.sh
```

### 3. Configure Environment

```bash
# Copy the production environment template
cp .env.production .env

# Edit the environment file with your settings
nano .env
```

**Important**: Update the following in your `.env` file:
- Database passwords
- Redis password
- Secret keys
- SMTP settings (if using email notifications)
- Twilio settings (if using SMS notifications)

### 4. Deploy the System

```bash
# Full deployment with monitoring
./deploy.sh deploy --with-monitoring

# Or basic deployment without monitoring
./deploy.sh deploy
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_PASSWORD` | Database password | - | ✅ |
| `REDIS_PASSWORD` | Redis password | - | ✅ |
| `SECRET_KEY` | Application secret key | - | ✅ |
| `DEMO_MODE` | Enable demo mode (no camera) | false | ❌ |
| `WORKERS` | Number of worker processes | 4 | ❌ |
| `LOG_LEVEL` | Logging level | info | ❌ |

### Camera Configuration

#### USB Camera
```bash
# Check available cameras
ls /dev/video*

# The system will automatically use /dev/video0
```

#### IP Camera
```bash
# Set camera URL in environment
echo "CAMERA_URL=rtsp://username:password@camera-ip:554/stream" >> .env
```

#### Demo Mode (No Camera)
```bash
# Enable demo mode
echo "DEMO_MODE=true" >> .env
```

## 🌐 Access Points

After successful deployment, the system will be available at:

- **Main Dashboard**: https://localhost (or your domain)
- **API Documentation**: https://localhost/docs
- **Health Check**: https://localhost/health
- **Metrics**: https://localhost/metrics (restricted access)

### Monitoring (if enabled)

- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin123` (change in production)

## 🔒 Security Considerations

### SSL Certificates

The deployment script generates self-signed certificates for development. For production:

1. **Obtain proper SSL certificates** from a Certificate Authority
2. **Replace the generated certificates**:
   ```bash
   cp your-cert.pem deployment/nginx/ssl/cert.pem
   cp your-key.pem deployment/nginx/ssl/key.pem
   ```
3. **Restart nginx**:
   ```bash
   docker-compose -f docker-compose.production.yml restart nginx
   ```

### Firewall Configuration

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block direct access to application port
sudo ufw deny 8080/tcp
```

### Password Security

- Use strong, unique passwords for all services
- Consider using a password manager
- Regularly rotate passwords

## 📊 Monitoring and Maintenance

### Health Checks

```bash
# Check system health
./deploy.sh health

# View application logs
./deploy.sh logs

# View specific service logs
./deploy.sh logs postgres
```

### Backup

```bash
# Create backup
docker exec shoplifting-postgres pg_dump -U shoplifter_user shoplifter_db > backup.sql

# Restore backup
docker exec -i shoplifting-postgres psql -U shoplifter_user shoplifter_db < backup.sql
```

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
./deploy.sh restart
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :8080

# Stop the conflicting service or change the port in .env
```

#### 2. Camera Not Detected
```bash
# Check camera permissions
ls -la /dev/video*

# Add user to video group
sudo usermod -a -G video $USER
```

#### 3. Database Connection Failed
```bash
# Check database logs
docker-compose -f docker-compose.production.yml logs postgres

# Verify database is running
docker-compose -f docker-compose.production.yml ps postgres
```

#### 4. High Memory Usage
```bash
# Check memory usage
docker stats

# Reduce worker count in .env
echo "WORKERS=2" >> .env
./deploy.sh restart
```

### Log Locations

- **Application logs**: `./logs/`
- **Nginx logs**: `./logs/nginx/`
- **Docker logs**: `docker-compose logs [service]`

## 🔄 Management Commands

```bash
# Start services
./deploy.sh start

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs [service]

# Check health
./deploy.sh health

# Complete cleanup (removes all data)
./deploy.sh cleanup
```

## 📈 Performance Optimization

### For High-Traffic Environments

1. **Increase worker processes**:
   ```bash
   echo "WORKERS=8" >> .env
   ```

2. **Enable Redis caching**:
   ```bash
   echo "ENABLE_CACHING=true" >> .env
   ```

3. **Optimize database**:
   ```bash
   # Increase shared_buffers in PostgreSQL
   # Add to docker-compose.yml postgres command:
   # -c shared_buffers=256MB -c max_connections=200
   ```

### For Resource-Constrained Environments

1. **Reduce worker processes**:
   ```bash
   echo "WORKERS=2" >> .env
   ```

2. **Disable monitoring**:
   ```bash
   # Deploy without --with-monitoring flag
   ./deploy.sh deploy
   ```

3. **Enable demo mode**:
   ```bash
   echo "DEMO_MODE=true" >> .env
   ```

## 🆘 Support

For issues and support:

1. Check the troubleshooting section above
2. Review application logs: `./deploy.sh logs`
3. Check system health: `./deploy.sh health`
4. Consult the API documentation at `/docs`

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.
