import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models.database import Alert, Event, get_db
from config import Config
import json


class AlertService:
    def __init__(self):
        self.alert_callbacks = []
        self.recent_alerts = {}  # person_id -> last_alert_time
        self.alert_cooldown = Config.ALERT_COOLDOWN

    def register_alert_callback(self, callback: Callable[[Dict], None]):
        """
        Register a callback function to be called when alerts are generated
        """
        self.alert_callbacks.append(callback)

    def unregister_alert_callback(self, callback: Callable[[Dict], None]):
        """
        Unregister an alert callback
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    async def create_alert(self, event_data: Dict, behavior_data: Dict,
                           anomaly_data: Dict = None) -> Optional[Dict]:
        """
        Create an alert based on event and behavior data
        """
        person_id = behavior_data.get('person_id')

        # Check cooldown period
        if self._is_in_cooldown(person_id):
            return None

        # Determine alert severity
        severity = self._calculate_severity(behavior_data, anomaly_data)

        # Create alert message
        alert_message = self._generate_alert_message(
            behavior_data, anomaly_data)

        # Store in database
        alert_data = await self._store_alert(event_data, behavior_data,
                                             severity, alert_message)

        if alert_data:
            # Update cooldown
            self.recent_alerts[person_id] = datetime.now()

            # Notify callbacks
            await self._notify_callbacks(alert_data)

            return alert_data

        return None

    def _is_in_cooldown(self, person_id: int) -> bool:
        """
        Check if person is in alert cooldown period
        """
        if person_id not in self.recent_alerts:
            return False

        last_alert_time = self.recent_alerts[person_id]
        time_since_last = (datetime.now() - last_alert_time).total_seconds()

        return time_since_last < self.alert_cooldown

    def _calculate_severity(self, behavior_data: Dict, anomaly_data: Dict = None) -> str:
        """
        Calculate alert severity based on behavior and anomaly data
        """
        suspicious_score = behavior_data.get('suspicious_score', 0)

        if anomaly_data:
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            combined_score = max(suspicious_score, anomaly_score)
        else:
            combined_score = suspicious_score

        if combined_score >= 0.9:
            return 'critical'
        elif combined_score >= 0.7:
            return 'high'
        elif combined_score >= 0.5:
            return 'medium'
        else:
            return 'low'

    def _generate_alert_message(self, behavior_data: Dict, anomaly_data: Dict = None) -> str:
        """
        Generate human-readable alert message focused on shoplifting detection
        """
        person_id = behavior_data.get('person_id')
        behaviors = behavior_data.get('behaviors', [])
        suspicious_score = behavior_data.get('suspicious_score', 0)
        shoplifting_stage = behavior_data.get('shoplifting_stage', 'browsing')

        # Check for shoplifting-specific behaviors
        shoplifting_behaviors = [b for b in behaviors if b['type'] in [
            'item_taking', 'concealment', 'shoplifting_intent', 'shelf_interaction']]

        if shoplifting_behaviors:
            # Generate shoplifting-specific alert message
            if shoplifting_stage == 'shoplifting':
                message_parts = [f"🚨 SHOPLIFTING ALERT - Person #{person_id}"]
                message_parts.append(
                    "CRITICAL: Person has concealed merchandise and is attempting to leave without payment")
                message_parts.append("IMMEDIATE ACTION REQUIRED")
            elif shoplifting_stage == 'concealing':
                message_parts = [
                    f"⚠️ CONCEALMENT DETECTED - Person #{person_id}"]
                message_parts.append(
                    "Person is hiding merchandise on their person")
                message_parts.append(
                    "Monitor closely - potential theft in progress")
            elif shoplifting_stage == 'taking':
                message_parts = [f"👀 ITEM REMOVAL - Person #{person_id}"]
                message_parts.append(
                    "Person has taken item from shelf/display")
                message_parts.append("Monitoring for concealment behavior")
            else:
                message_parts = [f"📍 SHELF INTERACTION - Person #{person_id}"]
                message_parts.append(
                    "Person is interacting with merchandise display")
        else:
            # Fallback for non-shoplifting alerts
            message_parts = [
                f"Suspicious activity detected for Person #{person_id}"]

        if behaviors:
            behavior_descriptions = [
                b.get('description', b.get('type', '')) for b in behaviors]
            message_parts.append("Detected behaviors:")
            message_parts.extend(
                [f"- {desc}" for desc in behavior_descriptions])

        message_parts.append(f"Confidence score: {suspicious_score:.2f}")

        if anomaly_data and anomaly_data.get('is_anomaly'):
            anomaly_score = anomaly_data.get('anomaly_score', 0)
            message_parts.append(
                f"Anomaly detection score: {anomaly_score:.2f}")

        return "\n".join(message_parts)

    async def _store_alert(self, event_data: Dict, behavior_data: Dict,
                           severity: str, message: str) -> Optional[Dict]:
        """
        Store alert in database
        """
        try:
            db = next(get_db())

            # Create event record first
            event = Event(
                event_type='suspicious_behavior',
                confidence=behavior_data.get('suspicious_score', 0),
                person_id=behavior_data.get('person_id'),
                x_coordinate=event_data.get('x_coordinate'),
                y_coordinate=event_data.get('y_coordinate'),
                width=event_data.get('width'),
                height=event_data.get('height'),
                description=json.dumps(behavior_data.get('behaviors', [])),
                image_path=event_data.get('image_path')
            )

            db.add(event)
            db.flush()  # Get the event ID

            # Create alert record
            alert = Alert(
                event_id=event.id,
                alert_type='suspicious_behavior',
                severity=severity,
                message=message
            )

            db.add(alert)
            db.commit()

            alert_data = {
                'id': alert.id,
                'event_id': event.id,
                'timestamp': alert.timestamp,
                'alert_type': alert.alert_type,
                'severity': severity,
                'message': message,
                'person_id': behavior_data.get('person_id'),
                'suspicious_score': behavior_data.get('suspicious_score', 0),
                'behaviors': behavior_data.get('behaviors', [])
            }

            db.close()
            return alert_data

        except Exception as e:
            print(f"Error storing alert: {e}")
            return None

    async def _notify_callbacks(self, alert_data: Dict):
        """
        Notify all registered callbacks about the alert
        """
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                print(f"Error in alert callback: {e}")

    async def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """
        Get recent alerts from database
        """
        try:
            db = next(get_db())

            alerts = db.query(Alert).order_by(
                Alert.timestamp.desc()).limit(limit).all()

            alert_list = []
            for alert in alerts:
                alert_dict = {
                    'id': alert.id,
                    'event_id': alert.event_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'acknowledged': alert.acknowledged,
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    'acknowledged_by': alert.acknowledged_by
                }
                alert_list.append(alert_dict)

            db.close()
            return alert_list

        except Exception as e:
            print(f"Error getting recent alerts: {e}")
            return []

    async def acknowledge_alert(self, alert_id: int, acknowledged_by: str = "system") -> bool:
        """
        Acknowledge an alert
        """
        try:
            db = next(get_db())

            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                db.commit()
                db.close()
                return True

            db.close()
            return False

        except Exception as e:
            print(f"Error acknowledging alert: {e}")
            return False

    def get_alert_stats(self) -> Dict:
        """
        Get alert statistics
        """
        try:
            db = next(get_db())

            # Get counts by severity
            total_alerts = db.query(Alert).count()
            critical_alerts = db.query(Alert).filter(
                Alert.severity == 'critical').count()
            high_alerts = db.query(Alert).filter(
                Alert.severity == 'high').count()
            medium_alerts = db.query(Alert).filter(
                Alert.severity == 'medium').count()
            low_alerts = db.query(Alert).filter(
                Alert.severity == 'low').count()

            # Get recent alerts (last 24 hours)
            yesterday = datetime.now() - timedelta(days=1)
            recent_alerts = db.query(Alert).filter(
                Alert.timestamp >= yesterday).count()

            # Get acknowledged alerts
            acknowledged_alerts = db.query(Alert).filter(
                Alert.acknowledged == True).count()

            db.close()

            return {
                'total_alerts': total_alerts,
                'critical_alerts': critical_alerts,
                'high_alerts': high_alerts,
                'medium_alerts': medium_alerts,
                'low_alerts': low_alerts,
                'recent_alerts_24h': recent_alerts,
                'acknowledged_alerts': acknowledged_alerts,
                'acknowledgment_rate': acknowledged_alerts / max(total_alerts, 1)
            }

        except Exception as e:
            print(f"Error getting alert stats: {e}")
            return {}
