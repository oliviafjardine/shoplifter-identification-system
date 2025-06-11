class SecurityDashboardApp {
    constructor() {
        this.ws = null;
        this.alerts = [];
        this.currentAlert = null;
        this.stats = {};
        this.systemStartTime = Date.now();

        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.loadInitialData();
        this.startPeriodicUpdates();
        this.updateTime();
        this.updateSystemUptime();
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'alert':
                this.handleNewAlert(message.data);
                break;
            case 'alert_acknowledged':
                this.handleAlertAcknowledged(message);
                break;
            case 'detection_update':
                this.handleDetectionUpdate(message.data);
                break;
            case 'pong':
                // Keep-alive response
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handleDetectionUpdate(data) {
        // Update people count in camera overlays
        const peopleCount = data.people_count || 0;

        // Update all camera people counts (for now just camera 1 since it's the only one connected)
        const peopleCountElement = document.getElementById('people-count-1');
        if (peopleCountElement) {
            peopleCountElement.textContent = peopleCount;
        }

        // Also update any other people count displays
        const allPeopleCountElements = document.querySelectorAll('[id^="people-count"]');
        allPeopleCountElements.forEach(element => {
            if (element.id === 'people-count-1') {
                element.textContent = peopleCount;
            } else {
                element.textContent = '0'; // Other cameras are not connected
            }
        });

        console.log(`People detection update: ${peopleCount} people detected`);
    }
    
    handleNewAlert(alertData) {
        // Add to alerts list
        this.alerts.unshift(alertData);
        
        // Limit alerts list size
        if (this.alerts.length > 100) {
            this.alerts = this.alerts.slice(0, 100);
        }
        
        // Update UI
        this.renderAlerts();
        this.showAlertModal(alertData);
        this.playAlertSound();
        
        // Update stats
        this.loadStats();
    }
    
    handleAlertAcknowledged(message) {
        if (message.success) {
            // Update alert in list
            const alert = this.alerts.find(a => a.id === message.alert_id);
            if (alert) {
                alert.acknowledged = true;
                this.renderAlerts();
            }
            
            // Close modal if it's the current alert
            if (this.currentAlert && this.currentAlert.id === message.alert_id) {
                this.closeAlertModal();
            }
        }
    }
    
    setupEventListeners() {
        // Alert controls
        document.getElementById('clear-alerts').addEventListener('click', () => {
            this.clearAlerts();
        });
        
        document.getElementById('refresh-alerts').addEventListener('click', () => {
            this.loadAlerts();
        });
        
        // Modal controls
        document.getElementById('acknowledge-alert').addEventListener('click', () => {
            this.acknowledgeCurrentAlert();
        });
        
        document.getElementById('dismiss-alert').addEventListener('click', () => {
            this.closeAlertModal();
        });
        
        document.querySelector('.close').addEventListener('click', () => {
            this.closeAlertModal();
        });
        
        // Close modal when clicking outside
        document.getElementById('alert-modal').addEventListener('click', (e) => {
            if (e.target.id === 'alert-modal') {
                this.closeAlertModal();
            }
        });
        
        // Camera control buttons
        this.setupCameraControls();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAlertModal();
                this.exitFullscreen();
            }
        });

        // Fullscreen change event
        document.addEventListener('fullscreenchange', () => {
            this.handleFullscreenChange();
        });
    }

    setupCameraControls() {
        // Setup expand buttons for fullscreen
        document.querySelectorAll('.control-btn').forEach(btn => {
            if (btn.querySelector('.fa-expand')) {
                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const cameraFeed = btn.closest('.camera-feed');
                    const cameraContent = cameraFeed.querySelector('.camera-content');
                    this.toggleFullscreen(cameraContent, cameraFeed);
                });
            }
        });
    }

    toggleFullscreen(element, cameraFeed) {
        if (!document.fullscreenElement) {
            this.enterFullscreen(element, cameraFeed);
        } else {
            this.exitFullscreen();
        }
    }

    enterFullscreen(element, cameraFeed) {
        const cameraTitle = cameraFeed.querySelector('.camera-title').textContent;

        // Add fullscreen class for styling
        element.classList.add('fullscreen-camera');

        // Create fullscreen overlay with camera info
        const overlay = document.createElement('div');
        overlay.className = 'fullscreen-overlay';
        overlay.innerHTML = `
            <div class="fullscreen-header">
                <h3>${cameraTitle}</h3>
                <div class="fullscreen-controls">
                    <button class="fullscreen-btn" id="exit-fullscreen">
                        <i class="fas fa-compress"></i> Exit Fullscreen
                    </button>
                </div>
            </div>
        `;

        element.appendChild(overlay);

        // Setup exit button
        overlay.querySelector('#exit-fullscreen').addEventListener('click', () => {
            this.exitFullscreen();
        });

        // Request fullscreen
        if (element.requestFullscreen) {
            element.requestFullscreen();
        } else if (element.webkitRequestFullscreen) {
            element.webkitRequestFullscreen();
        } else if (element.msRequestFullscreen) {
            element.msRequestFullscreen();
        }
    }

    exitFullscreen() {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
    }

    handleFullscreenChange() {
        if (!document.fullscreenElement) {
            // Remove fullscreen classes and overlays
            document.querySelectorAll('.fullscreen-camera').forEach(element => {
                element.classList.remove('fullscreen-camera');
                const overlay = element.querySelector('.fullscreen-overlay');
                if (overlay) {
                    overlay.remove();
                }
            });
        }
    }
    
    async loadInitialData() {
        await Promise.all([
            this.loadAlerts(),
            this.loadStats(),
            this.loadDetections()
        ]);
    }

    async loadDetections() {
        try {
            const response = await fetch('/api/detections');
            const data = await response.json();
            this.handleDetectionUpdate(data);
        } catch (error) {
            console.error('Error loading detections:', error);
        }
    }
    
    async loadAlerts() {
        try {
            const response = await fetch('/api/alerts');
            const data = await response.json();
            this.alerts = data.alerts || [];
            this.renderAlerts();
        } catch (error) {
            console.error('Error loading alerts:', error);
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch('/api/stats');
            this.stats = await response.json();
            this.renderStats();
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }
    
    renderAlerts() {
        const alertsList = document.getElementById('alerts-list');
        
        if (this.alerts.length === 0) {
            alertsList.innerHTML = `
                <div class="no-alerts">
                    <i class="fas fa-check-circle"></i>
                    <p>No alerts at this time</p>
                </div>
            `;
            return;
        }
        
        const alertsHtml = this.alerts.map(alert => this.createAlertHtml(alert)).join('');
        alertsList.innerHTML = alertsHtml;
        
        // Add event listeners to alert items
        alertsList.querySelectorAll('.alert-item').forEach(item => {
            item.addEventListener('click', () => {
                const alertId = parseInt(item.dataset.alertId);
                const alert = this.alerts.find(a => a.id === alertId);
                if (alert) {
                    this.showAlertModal(alert);
                }
            });
        });
    }
    
    createAlertHtml(alert) {
        // Parse the ISO timestamp properly and format it for display
        const date = new Date(alert.timestamp);
        const timestamp = date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'numeric',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });
        const acknowledgedClass = alert.acknowledged ? 'acknowledged' : '';

        return `
            <div class="alert-item ${alert.severity} ${acknowledgedClass}" data-alert-id="${alert.id}">
                <div class="alert-header">
                    <span class="alert-severity ${alert.severity}">${alert.severity}</span>
                    <span class="alert-time">${timestamp}</span>
                </div>
                <div class="alert-message">${this.formatAlertMessage(alert.message)}</div>
                <div class="alert-actions">
                    ${!alert.acknowledged ? `
                        <button class="btn btn-success btn-sm" onclick="app.acknowledgeAlert(${alert.id}); event.stopPropagation();">
                            <i class="fas fa-check"></i> Acknowledge
                        </button>
                    ` : `
                        <span class="acknowledged-badge">
                            <i class="fas fa-check-circle"></i> Acknowledged
                        </span>
                    `}
                </div>
            </div>
        `;
    }
    
    formatAlertMessage(message) {
        // Convert newlines to HTML breaks and format the message
        return message.replace(/\n/g, '<br>').replace(/- /g, '• ');
    }
    
    renderStats() {
        if (!this.stats.alerts) return;
        
        const alerts = this.stats.alerts;
        const camera = this.stats.camera || {};
        const tracking = this.stats.tracking || {};
        
        // Update status indicators
        document.getElementById('camera-status').textContent = camera.is_running ? 'Online' : 'Offline';
        document.getElementById('camera-status').className = `status-value ${camera.is_running ? 'online' : 'offline'}`;
        document.getElementById('active-tracks').textContent = tracking.active_tracks || 0;
        document.getElementById('alerts-today').textContent = alerts.recent_alerts_24h || 0;
        
        // Update stat cards
        document.getElementById('critical-alerts').textContent = alerts.critical_alerts || 0;
        document.getElementById('high-alerts').textContent = alerts.high_alerts || 0;
        document.getElementById('medium-alerts').textContent = alerts.medium_alerts || 0;
        
        const acknowledgmentRate = Math.round((alerts.acknowledgment_rate || 0) * 100);
        document.getElementById('acknowledged-rate').textContent = `${acknowledgmentRate}%`;
    }
    
    showAlertModal(alert) {
        this.currentAlert = alert;
        const modal = document.getElementById('alert-modal');
        const alertDetails = document.getElementById('alert-details');

        // Parse the ISO timestamp properly and format it for display
        const date = new Date(alert.timestamp);
        const timestamp = date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'numeric',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });
        const behaviors = alert.behaviors || [];
        
        let behaviorsList = '';
        if (behaviors.length > 0) {
            behaviorsList = `
                <div class="behavior-list">
                    <h4>Detected Behaviors:</h4>
                    <ul>
                        ${behaviors.map(b => `<li>${b.description || b.type} (${(b.confidence * 100).toFixed(1)}%)</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        alertDetails.innerHTML = `
            <div class="alert-modal-content">
                <div class="alert-modal-header">
                    <span class="alert-severity ${alert.severity}">${alert.severity.toUpperCase()}</span>
                    <span class="alert-timestamp">${timestamp}</span>
                </div>
                <div class="alert-modal-body">
                    <p><strong>Person ID:</strong> ${alert.person_id}</p>
                    <p><strong>Suspicion Score:</strong> ${(alert.suspicious_score * 100).toFixed(1)}%</p>
                    ${behaviorsList}
                    <div class="alert-message-full">
                        <h4>Alert Message:</h4>
                        <p>${this.formatAlertMessage(alert.message)}</p>
                    </div>
                </div>
            </div>
        `;
        
        // Show/hide acknowledge button based on alert status
        const acknowledgeBtn = document.getElementById('acknowledge-alert');
        acknowledgeBtn.style.display = alert.acknowledged ? 'none' : 'inline-flex';
        
        modal.style.display = 'block';
    }
    
    closeAlertModal() {
        document.getElementById('alert-modal').style.display = 'none';
        this.currentAlert = null;
    }
    
    async acknowledgeAlert(alertId) {
        try {
            const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });
            
            if (response.ok) {
                // Update local alert
                const alert = this.alerts.find(a => a.id === alertId);
                if (alert) {
                    alert.acknowledged = true;
                    this.renderAlerts();
                }
                
                // Close modal if it's the current alert
                if (this.currentAlert && this.currentAlert.id === alertId) {
                    this.closeAlertModal();
                }
                
                this.showNotification('Alert acknowledged successfully', 'success');
            } else {
                this.showNotification('Failed to acknowledge alert', 'error');
            }
        } catch (error) {
            console.error('Error acknowledging alert:', error);
            this.showNotification('Error acknowledging alert', 'error');
        }
    }
    
    async acknowledgeCurrentAlert() {
        if (this.currentAlert) {
            await this.acknowledgeAlert(this.currentAlert.id);
        }
    }
    
    clearAlerts() {
        if (confirm('Are you sure you want to clear all alerts?')) {
            this.alerts = [];
            this.renderAlerts();
            this.showNotification('Alerts cleared', 'info');
        }
    }
    
    playAlertSound() {
        // Create a simple beep sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    }
    
    showNotification(message, type = 'info') {
        // Simple notification system
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        
        switch (type) {
            case 'success':
                notification.style.backgroundColor = '#27ae60';
                break;
            case 'error':
                notification.style.backgroundColor = '#e74c3c';
                break;
            case 'warning':
                notification.style.backgroundColor = '#f39c12';
                break;
            default:
                notification.style.backgroundColor = '#3498db';
        }
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    updateConnectionStatus(connected) {
        const statusElements = document.querySelectorAll('.connection-status');
        statusElements.forEach(el => {
            el.textContent = connected ? 'Connected' : 'Disconnected';
            el.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
        });
    }
    
    startPeriodicUpdates() {
        // Update stats every 30 seconds
        setInterval(() => {
            this.loadStats();
        }, 30000);

        // Send keep-alive ping every 30 seconds
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);

        // Update time every second
        setInterval(() => {
            this.updateTime();
        }, 1000);

        // Update uptime every minute
        setInterval(() => {
            this.updateSystemUptime();
        }, 60000);

        // Update system monitoring every 5 seconds
        setInterval(() => {
            this.updateSystemMonitoring();
        }, 5000);
    }

    updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: true,
            hour: 'numeric',
            minute: '2-digit'
        });
        const dateString = now.toLocaleDateString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });

        const timeElement = document.getElementById('current-time');
        const dateElement = document.getElementById('current-date');

        if (timeElement) timeElement.textContent = timeString;
        if (dateElement) dateElement.textContent = dateString;
    }

    updateSystemUptime() {
        const uptime = Date.now() - this.systemStartTime;
        const days = Math.floor(uptime / (1000 * 60 * 60 * 24));
        const hours = Math.floor((uptime % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));

        const uptimeString = `${days}d ${hours}h ${minutes}m`;
        const uptimeElement = document.getElementById('system-uptime');

        if (uptimeElement) uptimeElement.textContent = uptimeString;
    }

    updateSystemMonitoring() {
        // Simulate system monitoring data
        const cpuUsage = Math.floor(Math.random() * 30) + 10; // 10-40%
        const memoryUsage = Math.floor(Math.random() * 20) + 35; // 35-55%
        const systemLoad = Math.max(cpuUsage, memoryUsage);

        const storageUsed = 2.66 + (Math.random() * 0.5); // 2.66-3.16 GB
        const storagePercent = Math.floor((storageUsed / 7.64) * 100);

        // Update system load
        const systemLoadElement = document.getElementById('system-load');
        const cpuElement = document.getElementById('cpu-usage');
        const memoryElement = document.getElementById('memory-usage');

        if (systemLoadElement) systemLoadElement.textContent = `${systemLoad}%`;
        if (cpuElement) cpuElement.textContent = `${cpuUsage}%`;
        if (memoryElement) memoryElement.textContent = `${memoryUsage}%`;

        // Update progress circle
        const progressCircle = document.querySelector('.progress-circle');
        if (progressCircle) {
            const gradient = `conic-gradient(var(--accent-warning) ${systemLoad}%, var(--border-color) ${systemLoad}%)`;
            progressCircle.style.background = gradient;
        }



        // Update storage
        const storageUsedElement = document.getElementById('storage-used');
        const storagePercentElement = document.getElementById('storage-percent');
        const storageProgress = document.querySelector('.storage-progress');

        if (storageUsedElement) storageUsedElement.textContent = `${storageUsed.toFixed(2)} GB`;
        if (storagePercentElement) storagePercentElement.textContent = `${storagePercent}%`;
        if (storageProgress) storageProgress.style.width = `${storagePercent}%`;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SecurityDashboardApp();
});

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .acknowledged-badge {
        color: #27ae60;
        font-weight: 500;
    }
    
    .alert-item.acknowledged {
        opacity: 0.7;
    }
    
    .behavior-list ul {
        margin: 10px 0;
        padding-left: 20px;
    }
    
    .behavior-list li {
        margin: 5px 0;
    }
    
    .alert-modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }
    
    .alert-timestamp {
        font-size: 0.9rem;
        color: #666;
    }
    
    .alert-message-full {
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #eee;
    }
    
    .btn-sm {
        padding: 4px 8px;
        font-size: 0.8rem;
    }
`;
document.head.appendChild(style);
