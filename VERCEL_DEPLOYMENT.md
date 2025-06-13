# 🚀 Vercel Deployment Guide

## Shoplifting Detection System - Vercel Deployment

This guide explains how to deploy your shoplifting detection system to Vercel in demo mode.

## 🎭 Demo Mode Features

Since Vercel is a serverless platform, the application runs in **demo mode** with the following features:

- ✅ **Professional Dashboard UI** - Full web interface
- ✅ **API Documentation** - FastAPI docs at `/docs`
- ✅ **System Statistics** - Real-time stats at `/api/stats`
- ✅ **Health Checks** - Monitoring endpoints
- ✅ **Demo Video Feed** - Simulated camera feed
- ❌ **Live Camera** - Not available in serverless environment
- ❌ **ML Model Training** - Heavy ML libraries excluded for performance

## 📁 Files for Vercel Deployment

### Required Files
- ✅ `main.py` - Modified for serverless compatibility
- ✅ `vercel.json` - Vercel configuration
- ✅ `requirements-vercel.txt` - Lightweight dependencies

### Key Modifications Made
1. **Conditional imports** - OpenCV/ML libraries only imported if available
2. **Demo mode detection** - Automatically enables when CV libraries unavailable
3. **Simplified video feed** - Uses PIL for demo frames instead of camera
4. **Lightweight dependencies** - Reduced from 72 to 15 packages

## 🚀 Deployment Steps

### 1. Connect to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up/login with GitHub
3. Click "New Project"
4. Import your repository

### 2. Configure Project
- **Framework Preset**: Other
- **Root Directory**: `./` (leave default)
- **Build Command**: (leave empty)
- **Output Directory**: (leave empty)
- **Install Command**: `pip install -r requirements-vercel.txt`

### 3. Environment Variables (Optional)
Add these in Vercel dashboard if needed:
- `DEMO_MODE=true`
- `ENVIRONMENT=production`

### 4. Deploy
Click "Deploy" and wait for build to complete.

## 🌐 Access Your Deployed App

After deployment, you'll get a URL like: `https://your-app-name.vercel.app`

### Available Endpoints
- **Dashboard**: `https://your-app.vercel.app/`
- **API Docs**: `https://your-app.vercel.app/docs`
- **Health Check**: `https://your-app.vercel.app/health`
- **Statistics**: `https://your-app.vercel.app/api/stats`
- **Video Feed**: `https://your-app.vercel.app/video_feed` (demo mode)

## 🔧 Troubleshooting

### Common Issues

#### 1. Build Timeout
- **Cause**: Heavy dependencies taking too long
- **Solution**: Use `requirements-vercel.txt` (already configured)

#### 2. Function Size Limit
- **Cause**: Package size too large
- **Solution**: Reduced dependencies in vercel config

#### 3. Cold Start Issues
- **Cause**: Serverless functions sleeping
- **Solution**: Expected behavior, first request may be slow

#### 4. Video Feed Not Working
- **Cause**: Serverless limitations
- **Solution**: Demo mode provides simulated feed

### Performance Notes
- **Cold starts**: First request after inactivity may take 5-10 seconds
- **Function timeout**: 30 seconds max execution time
- **Memory limit**: 1GB max memory usage
- **No persistent storage**: Files don't persist between requests

## 🔄 Local vs Vercel Differences

| Feature | Local Development | Vercel Deployment |
|---------|------------------|-------------------|
| Camera Access | ✅ Real camera | ❌ Demo mode only |
| ML Training | ✅ Full training | ❌ Not available |
| File Storage | ✅ Persistent | ❌ Temporary only |
| Background Tasks | ✅ Threading | ❌ Limited |
| Dependencies | 72 packages | 15 packages |
| Startup Time | ~5 seconds | ~10-30 seconds |

## 📈 Next Steps

### For Production Use
If you need full functionality (camera access, ML training), consider:

1. **Traditional VPS/Cloud Server** - Use the Docker deployment
2. **AWS/GCP/Azure** - Use container services
3. **Hybrid Approach** - Vercel for dashboard, separate server for processing

### For Demo/Presentation
The Vercel deployment is perfect for:
- ✅ Showcasing the UI/UX
- ✅ Demonstrating API functionality
- ✅ Portfolio/presentation purposes
- ✅ Client demos without hardware setup

## 🆘 Support

If you encounter issues:
1. Check Vercel function logs in dashboard
2. Verify `requirements-vercel.txt` is being used
3. Ensure `DEMO_MODE=true` is set
4. Check the health endpoint: `/health`

The demo mode provides a fully functional web interface that showcases your professional surveillance system design!
