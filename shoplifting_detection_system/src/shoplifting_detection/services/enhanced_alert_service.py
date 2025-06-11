"""
Enhanced Alert Service for Shoplifting Detection System
Implements comprehensive alert management with evidence packages and multi-channel notifications
Supports REQ-007, REQ-008, REQ-046: Advanced alert management and notification system
"""

import asyncio
import json
import uuid
import os
import cv2
import smtplib
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import requests
from sqlalchemy.orm import Session

from models.database import Alert, Event, Camera, Person, AuditLog, get_db
from config import Config, AlertSeverity, DetectionBehavior

logger = logging.getLogger(__name__)


class EnhancedAlertService:
    """
    Enhanced Alert Service implementing comprehensive alert management
    - REQ-007: Structured alerts with evidence packages
    - REQ-008: Multiple notification channels
    - REQ-046: Sophisticated alert management and escalation
    """

    def __init__(self):
        self.alert_callbacks = []
        self.recent_alerts = {}  # person_id -> last_alert_time
        self.alert_cooldown = Config.ALERT_COOLDOWN

        # Notification channels
        self.notification_channels = {
            'email': self._send_email_notification,
            'sms': self._send_sms_notification,
            'push': self._send_push_notification,
            'webhook': self._send_webhook_notification
        }

        # Evidence storage
        self.evidence_base_path = "evidence"
        os.makedirs(self.evidence_base_path, exist_ok=True)

        # Escalation configuration
        self.escalation_rules = {
            AlertSeverity.CRITICAL: {
                'immediate_channels': ['push', 'sms', 'email'],
                'escalation_delay': 60,  # seconds
                'max_escalations': 3
            },
            AlertSeverity.HIGH: {
                'immediate_channels': ['push', 'email'],
                'escalation_delay': 300,  # 5 minutes
                'max_escalations': 2
            },
            AlertSeverity.MEDIUM: {
                'immediate_channels': ['push'],
                'escalation_delay': 900,  # 15 minutes
                'max_escalations': 1
            },
            AlertSeverity.LOW: {
                'immediate_channels': [],
                'escalation_delay': 3600,  # 1 hour
                'max_escalations': 0
            }
        }

        logger.info("EnhancedAlertService initialized")

    async def create_comprehensive_alert(self, detection_result: Dict,
                                         camera_id: str, frame: Any = None) -> Optional[str]:
        """
        Create comprehensive alert with evidence package (REQ-007)

        Args:
            detection_result: Detection result from ensemble detector
            camera_id: Camera identifier
            frame: Current video frame for evidence

        Returns:
            Alert UUID if successful, None otherwise
        """
        try:
            # Check cooldown
            person_id = detection_result['person_id']
            if self._is_in_cooldown(person_id):
                return None

            db = next(get_db())

            # Generate evidence package
            evidence_package = await self._create_evidence_package(
                detection_result, camera_id, frame
            )

            # Create event record with enhanced metadata
            event = Event(
                event_uuid=uuid.uuid4(),
                event_type="shoplifting_detection",
                behavior_type=detection_result['behavior_type'].value,
                confidence=detection_result['confidence'],
                severity=detection_result['severity'].value,
                camera_id=self._get_camera_db_id(camera_id, db),
                person_id=self._get_or_create_person_id(
                    detection_result['person_id'], db),
                x_coordinate=detection_result['bounding_box']['x'],
                y_coordinate=detection_result['bounding_box']['y'],
                width=detection_result['bounding_box']['width'],
                height=detection_result['bounding_box']['height'],
                description=self._generate_enhanced_description(
                    detection_result),
                metadata=detection_result['evidence'],
                video_clip_path=evidence_package.get('video_path'),
                image_path=evidence_package.get('image_path'),
                evidence_package_path=evidence_package.get('package_path'),
                processing_time_ms=detection_result['processing_time_ms'],
                model_version=json.dumps(detection_result['model_versions'])
            )

            db.add(event)
            db.commit()
            db.refresh(event)

            # Create comprehensive alert
            alert = Alert(
                alert_uuid=uuid.uuid4(),
                event_id=event.id,
                alert_type=detection_result['behavior_type'].value,
                severity=detection_result['severity'].value,
                priority_score=self._calculate_priority_score(
                    detection_result),
                title=self._generate_alert_title(detection_result),
                message=self._generate_comprehensive_message(
                    detection_result, event),
                recommendation=self._generate_recommendation(detection_result),
                notifications_sent={}
            )

            db.add(alert)
            db.commit()
            db.refresh(alert)

            # Log audit trail
            await self._log_alert_creation(alert, event, db)

            # Update cooldown
            self.recent_alerts[person_id] = datetime.now()

            # Send notifications based on severity (REQ-008)
            await self._process_alert_notifications(alert, event, db)

            # Notify callbacks
            await self._notify_callbacks({
                'alert_uuid': str(alert.alert_uuid),
                'severity': alert.severity,
                'behavior_type': detection_result['behavior_type'].value,
                'confidence': detection_result['confidence'],
                'person_id': person_id,
                'camera_id': camera_id,
                'timestamp': event.timestamp.isoformat(),
                'evidence_package': evidence_package
            })

            db.close()

            logger.info(f"Comprehensive alert created: {alert.alert_uuid}")
            return str(alert.alert_uuid)

        except Exception as e:
            logger.error(f"Error creating comprehensive alert: {e}")
            return None

    async def _create_evidence_package(self, detection_result: Dict,
                                       camera_id: str, frame: Any) -> Dict[str, str]:
        """Create comprehensive evidence package"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            person_id = detection_result['person_id']

            # Create evidence directory
            evidence_dir = os.path.join(
                self.evidence_base_path,
                f"{timestamp}_{camera_id}_{person_id}"
            )
            os.makedirs(evidence_dir, exist_ok=True)

            evidence_package = {}

            # Save key frame image
            if frame is not None:
                image_path = os.path.join(evidence_dir, "key_frame.jpg")
                cv2.imwrite(image_path, frame)
                evidence_package['image_path'] = image_path

            # Create video clip placeholder (would need video buffer implementation)
            video_path = os.path.join(evidence_dir, "video_clip.mp4")
            evidence_package['video_path'] = video_path

            # Create evidence metadata file
            metadata_path = os.path.join(evidence_dir, "metadata.json")
            metadata = {
                'detection_result': {
                    'behavior_type': detection_result['behavior_type'].value,
                    'confidence': detection_result['confidence'],
                    'severity': detection_result['severity'].value,
                    'bounding_box': detection_result['bounding_box'],
                    'processing_time_ms': detection_result['processing_time_ms'],
                    'model_versions': detection_result['model_versions']
                },
                'camera_info': {
                    'camera_id': camera_id,
                    'timestamp': timestamp
                },
                'evidence_files': {
                    'key_frame': 'key_frame.jpg',
                    'video_clip': 'video_clip.mp4'
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            evidence_package['package_path'] = evidence_dir
            evidence_package['metadata_path'] = metadata_path

            return evidence_package

        except Exception as e:
            logger.error(f"Error creating evidence package: {e}")
            return {}

    def _calculate_priority_score(self, detection_result: Dict) -> float:
        """Calculate priority score for alert ordering"""
        base_score = detection_result['confidence']

        # Adjust based on behavior type
        behavior_weights = {
            DetectionBehavior.SECURITY_TAG_REMOVAL: 1.2,
            DetectionBehavior.COORDINATED_THEFT: 1.15,
            DetectionBehavior.ITEM_CONCEALMENT: 1.1,
            DetectionBehavior.POCKET_STUFFING: 1.05,
            DetectionBehavior.BAG_LOADING: 1.0,
            DetectionBehavior.PRICE_TAG_SWITCHING: 0.95,
            DetectionBehavior.MULTIPLE_ITEM_HANDLING: 0.9,
            DetectionBehavior.EXIT_WITHOUT_PAYMENT: 1.3
        }

        behavior_type = detection_result['behavior_type']
        weight = behavior_weights.get(behavior_type, 1.0)

        return min(base_score * weight, 1.0)

    def _generate_alert_title(self, detection_result: Dict) -> str:
        """Generate descriptive alert title"""
        behavior_type = detection_result['behavior_type']
        confidence = detection_result['confidence']
        severity = detection_result['severity']

        behavior_names = {
            DetectionBehavior.ITEM_CONCEALMENT: "Item Concealment",
            DetectionBehavior.SECURITY_TAG_REMOVAL: "Security Tag Removal",
            DetectionBehavior.POCKET_STUFFING: "Pocket Stuffing",
            DetectionBehavior.BAG_LOADING: "Bag Loading",
            DetectionBehavior.COORDINATED_THEFT: "Coordinated Theft",
            DetectionBehavior.PRICE_TAG_SWITCHING: "Price Tag Switching",
            DetectionBehavior.EXIT_WITHOUT_PAYMENT: "Exit Without Payment",
            DetectionBehavior.MULTIPLE_ITEM_HANDLING: "Multiple Item Handling"
        }

        behavior_name = behavior_names.get(
            behavior_type, "Suspicious Behavior")

        return f"{severity.value.upper()}: {behavior_name} Detected ({confidence:.1%} confidence)"

    def _generate_comprehensive_message(self, detection_result: Dict, event: Event) -> str:
        """Generate comprehensive alert message"""
        behavior_type = detection_result['behavior_type']
        confidence = detection_result['confidence']
        person_id = detection_result['person_id']

        message = f"Shoplifting behavior detected:\n\n"
        message += f"• Behavior: {behavior_type.value.replace('_', ' ').title()}\n"
        message += f"• Confidence: {confidence:.1%}\n"
        message += f"• Person ID: {person_id}\n"
        message += f"• Location: Camera {event.camera_id}\n"
        message += f"• Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"• Processing Time: {detection_result['processing_time_ms']:.1f}ms\n\n"

        # Add evidence information
        if event.video_clip_path:
            message += f"• Video Evidence: Available\n"
        if event.image_path:
            message += f"• Image Evidence: Available\n"

        return message

    def _generate_recommendation(self, detection_result: Dict) -> str:
        """Generate recommended actions based on detection"""
        behavior_type = detection_result['behavior_type']
        severity = detection_result['severity']

        recommendations = {
            DetectionBehavior.ITEM_CONCEALMENT: "Approach customer discreetly and offer assistance. Monitor closely.",
            DetectionBehavior.SECURITY_TAG_REMOVAL: "Immediate intervention required. Contact security personnel.",
            DetectionBehavior.POCKET_STUFFING: "Monitor customer movement. Prepare for potential intervention.",
            DetectionBehavior.BAG_LOADING: "Observe customer at checkout. Verify all items are paid for.",
            DetectionBehavior.COORDINATED_THEFT: "Alert all security personnel. Monitor all involved individuals.",
            DetectionBehavior.PRICE_TAG_SWITCHING: "Check item pricing at checkout. Verify tag authenticity.",
            DetectionBehavior.EXIT_WITHOUT_PAYMENT: "Immediate intervention at exit. Check receipt.",
            DetectionBehavior.MULTIPLE_ITEM_HANDLING: "Monitor customer behavior. Offer shopping assistance."
        }

        base_recommendation = recommendations.get(
            behavior_type, "Monitor customer behavior closely.")

        if severity == AlertSeverity.CRITICAL:
            return f"URGENT: {base_recommendation} Consider immediate security response."
        elif severity == AlertSeverity.HIGH:
            return f"HIGH PRIORITY: {base_recommendation} Escalate if behavior continues."
        else:
            return base_recommendation

    async def _process_alert_notifications(self, alert: Alert, event: Event, db: Session):
        """Process notifications based on alert severity and escalation rules"""
        try:
            severity = AlertSeverity(alert.severity)
            rules = self.escalation_rules.get(severity, {})
            immediate_channels = rules.get('immediate_channels', [])

            notifications_sent = {}

            # Send immediate notifications
            for channel in immediate_channels:
                if channel in self.notification_channels:
                    try:
                        success = await self.notification_channels[channel](alert, event)
                        notifications_sent[channel] = {
                            'sent_at': datetime.now().isoformat(),
                            'success': success
                        }
                    except Exception as e:
                        logger.error(
                            f"Error sending {channel} notification: {e}")
                        notifications_sent[channel] = {
                            'sent_at': datetime.now().isoformat(),
                            'success': False,
                            'error': str(e)
                        }

            # Update alert with notification status
            alert.notifications_sent = notifications_sent
            db.commit()

            # Schedule escalation if needed
            if rules.get('max_escalations', 0) > 0:
                await self._schedule_escalation(alert, event, rules)

        except Exception as e:
            logger.error(f"Error processing alert notifications: {e}")

    async def _send_email_notification(self, alert: Alert, event: Event) -> bool:
        """Send email notification"""
        try:
            # Email configuration (would be in environment variables)
            smtp_server = os.getenv('SMTP_SERVER', 'localhost')
            smtp_port = int(os.getenv('SMTP_PORT', 587))
            smtp_username = os.getenv('SMTP_USERNAME', '')
            smtp_password = os.getenv('SMTP_PASSWORD', '')
            from_email = os.getenv(
                'FROM_EMAIL', 'alerts@shoplifting-detection.com')
            to_emails = os.getenv('ALERT_EMAILS', '').split(',')

            if not to_emails or not to_emails[0]:
                logger.warning("No email recipients configured")
                return False

            # Create message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = alert.title

            # Email body
            body = f"""
{alert.message}

Recommendation: {alert.recommendation}

Alert Details:
- Alert ID: {alert.alert_uuid}
- Severity: {alert.severity}
- Priority Score: {alert.priority_score}
- Timestamp: {alert.timestamp}

Event Details:
- Event ID: {event.event_uuid}
- Camera: {event.camera_id}
- Confidence: {event.confidence:.1%}
- Processing Time: {event.processing_time_ms}ms

Evidence:
- Image: {event.image_path or 'Not available'}
- Video: {event.video_clip_path or 'Not available'}
- Package: {event.evidence_package_path or 'Not available'}
"""

            msg.attach(MimeText(body, 'plain'))

            # Attach evidence image if available
            if event.image_path and os.path.exists(event.image_path):
                with open(event.image_path, 'rb') as f:
                    part = MimeBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= "evidence_{alert.alert_uuid}.jpg"'
                    )
                    msg.attach(part)

            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)

            text = msg.as_string()
            server.sendmail(from_email, to_emails, text)
            server.quit()

            logger.info(
                f"Email notification sent for alert {alert.alert_uuid}")
            return True

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False

    async def _send_sms_notification(self, alert: Alert, event: Event) -> bool:
        """Send SMS notification"""
        try:
            # SMS configuration (using Twilio or similar service)
            sms_api_url = os.getenv('SMS_API_URL', '')
            sms_api_key = os.getenv('SMS_API_KEY', '')
            sms_numbers = os.getenv('ALERT_SMS_NUMBERS', '').split(',')

            if not sms_api_url or not sms_numbers or not sms_numbers[0]:
                logger.warning("SMS notification not configured")
                return False

            # Create SMS message
            message = f"SHOPLIFTING ALERT: {alert.severity.upper()} - {alert.alert_type} detected. "
            message += f"Person {event.person_id} at Camera {event.camera_id}. "
            message += f"Confidence: {event.confidence:.1%}. Check dashboard for details."

            # Send SMS to each number
            success_count = 0
            for number in sms_numbers:
                if number.strip():
                    try:
                        response = requests.post(sms_api_url, {
                            'api_key': sms_api_key,
                            'to': number.strip(),
                            'message': message
                        }, timeout=10)

                        if response.status_code == 200:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error sending SMS to {number}: {e}")

            success = success_count > 0
            if success:
                logger.info(
                    f"SMS notification sent for alert {alert.alert_uuid}")

            return success

        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            return False

    async def _send_push_notification(self, alert: Alert, event: Event) -> bool:
        """Send push notification"""
        try:
            # Push notification configuration
            push_api_url = os.getenv('PUSH_API_URL', '')
            push_api_key = os.getenv('PUSH_API_KEY', '')

            if not push_api_url:
                logger.warning("Push notification not configured")
                return False

            # Create push notification payload
            payload = {
                'title': alert.title,
                'body': f"Person {event.person_id} - {alert.alert_type}",
                'data': {
                    'alert_uuid': str(alert.alert_uuid),
                    'severity': alert.severity,
                    'camera_id': event.camera_id,
                    'confidence': event.confidence
                }
            }

            headers = {
                'Authorization': f'Bearer {push_api_key}',
                'Content-Type': 'application/json'
            }

            response = requests.post(
                push_api_url, json=payload, headers=headers, timeout=10)

            success = response.status_code == 200
            if success:
                logger.info(
                    f"Push notification sent for alert {alert.alert_uuid}")

            return success

        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return False

    async def _send_webhook_notification(self, alert: Alert, event: Event) -> bool:
        """Send webhook notification"""
        try:
            webhook_url = os.getenv('WEBHOOK_URL', '')

            if not webhook_url:
                return False

            # Create webhook payload
            payload = {
                'alert': {
                    'uuid': str(alert.alert_uuid),
                    'title': alert.title,
                    'message': alert.message,
                    'severity': alert.severity,
                    'priority_score': alert.priority_score,
                    'timestamp': alert.timestamp.isoformat()
                },
                'event': {
                    'uuid': str(event.event_uuid),
                    'behavior_type': event.behavior_type,
                    'confidence': event.confidence,
                    'camera_id': event.camera_id,
                    'person_id': event.person_id,
                    'bounding_box': {
                        'x': event.x_coordinate,
                        'y': event.y_coordinate,
                        'width': event.width,
                        'height': event.height
                    }
                }
            }

            response = requests.post(webhook_url, json=payload, timeout=10)

            success = response.status_code == 200
            if success:
                logger.info(
                    f"Webhook notification sent for alert {alert.alert_uuid}")

            return success

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False

    def _get_camera_db_id(self, camera_id: str, db: Session) -> int:
        """Get or create camera database ID"""
        try:
            camera = db.query(Camera).filter(
                Camera.camera_id == camera_id).first()
            if not camera:
                camera = Camera(
                    camera_id=camera_id,
                    name=f"Camera {camera_id}",
                    status="active"
                )
                db.add(camera)
                db.commit()
                db.refresh(camera)
            return camera.id
        except Exception as e:
            logger.error(f"Error getting camera DB ID: {e}")
            return 1  # Default camera ID

    def _get_or_create_person_id(self, person_id: int, db: Session) -> int:
        """Get or create person database ID"""
        try:
            person = db.query(Person).filter(Person.id == person_id).first()
            if not person:
                person = Person(
                    id=person_id,
                    person_uuid=uuid.uuid4(),
                    status="active"
                )
                db.add(person)
                db.commit()
                db.refresh(person)
            return person.id
        except Exception as e:
            logger.error(f"Error getting person DB ID: {e}")
            return person_id

    def _generate_enhanced_description(self, detection_result: Dict) -> str:
        """Generate enhanced description for the event"""
        behavior_type = detection_result['behavior_type']
        confidence = detection_result['confidence']

        descriptions = {
            DetectionBehavior.ITEM_CONCEALMENT: f"Person concealing items with {confidence:.1%} confidence",
            DetectionBehavior.SECURITY_TAG_REMOVAL: f"Security tag removal detected with {confidence:.1%} confidence",
            DetectionBehavior.POCKET_STUFFING: f"Items being placed in pockets with {confidence:.1%} confidence",
            DetectionBehavior.BAG_LOADING: f"Items being loaded into bag with {confidence:.1%} confidence",
            DetectionBehavior.COORDINATED_THEFT: f"Coordinated theft behavior with {confidence:.1%} confidence",
            DetectionBehavior.PRICE_TAG_SWITCHING: f"Price tag manipulation with {confidence:.1%} confidence",
            DetectionBehavior.EXIT_WITHOUT_PAYMENT: f"Attempting to exit without payment with {confidence:.1%} confidence",
            DetectionBehavior.MULTIPLE_ITEM_HANDLING: f"Suspicious item handling with {confidence:.1%} confidence"
        }

        return descriptions.get(behavior_type, f"Suspicious behavior detected with {confidence:.1%} confidence")

    async def _log_alert_creation(self, alert: Alert, event: Event, db: Session):
        """Log alert creation for audit trail"""
        try:
            audit_log = AuditLog(
                user_id="system",
                action="alert_created",
                resource_type="alert",
                resource_id=str(alert.alert_uuid),
                action_details={
                    "alert_severity": alert.severity,
                    "alert_type": alert.alert_type,
                    "event_uuid": str(event.event_uuid),
                    "confidence": event.confidence,
                    "camera_id": event.camera_id,
                    "person_id": event.person_id
                },
                success=True
            )
            db.add(audit_log)
            db.commit()
        except Exception as e:
            logger.error(f"Error logging alert creation: {e}")

    async def _schedule_escalation(self, alert: Alert, event: Event, rules: Dict):
        """Schedule alert escalation"""
        # This would implement escalation logic
        # For now, just log the escalation schedule
        escalation_delay = rules.get('escalation_delay', 300)
        max_escalations = rules.get('max_escalations', 1)

        logger.info(f"Escalation scheduled for alert {alert.alert_uuid} "
                    f"in {escalation_delay} seconds (max: {max_escalations})")

    def _is_in_cooldown(self, person_id: int) -> bool:
        """Check if person is in alert cooldown period"""
        if person_id not in self.recent_alerts:
            return False

        last_alert_time = self.recent_alerts[person_id]
        time_since_last = (datetime.now() - last_alert_time).total_seconds()

        return time_since_last < self.alert_cooldown

    def register_alert_callback(self, callback: Callable[[Dict], None]):
        """Register a callback function to be called when alerts are generated"""
        self.alert_callbacks.append(callback)

    def unregister_alert_callback(self, callback: Callable[[Dict], None]):
        """Unregister an alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    async def _notify_callbacks(self, alert_data: Dict):
        """Notify all registered callbacks about the alert"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def acknowledge_alert(self, alert_uuid: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            db = next(get_db())

            alert = db.query(Alert).filter(
                Alert.alert_uuid == alert_uuid).first()
            if alert:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                alert.status = "acknowledged"
                db.commit()

                # Log acknowledgment
                audit_log = AuditLog(
                    user_id=acknowledged_by,
                    action="alert_acknowledged",
                    resource_type="alert",
                    resource_id=alert_uuid,
                    action_details={
                        "acknowledged_at": datetime.now().isoformat()},
                    success=True
                )
                db.add(audit_log)
                db.commit()

                db.close()
                return True

            db.close()
            return False

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False

    async def resolve_alert(self, alert_uuid: str, resolved_by: str,
                            resolution_notes: str = "") -> bool:
        """Resolve an alert"""
        try:
            db = next(get_db())

            alert = db.query(Alert).filter(
                Alert.alert_uuid == alert_uuid).first()
            if alert:
                alert.status = "resolved"
                alert.resolved_at = datetime.now()
                alert.resolved_by = resolved_by
                alert.resolution_notes = resolution_notes
                db.commit()

                # Log resolution
                audit_log = AuditLog(
                    user_id=resolved_by,
                    action="alert_resolved",
                    resource_type="alert",
                    resource_id=alert_uuid,
                    action_details={
                        "resolved_at": datetime.now().isoformat(),
                        "resolution_notes": resolution_notes
                    },
                    success=True
                )
                db.add(audit_log)
                db.commit()

                db.close()
                return True

            db.close()
            return False

        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False

    async def mark_false_positive(self, alert_uuid: str, marked_by: str,
                                  notes: str = "") -> bool:
        """Mark alert as false positive"""
        try:
            db = next(get_db())

            alert = db.query(Alert).filter(
                Alert.alert_uuid == alert_uuid).first()
            if alert:
                alert.status = "false_positive"
                alert.resolved_at = datetime.now()
                alert.resolved_by = marked_by
                alert.resolution_notes = f"False positive: {notes}"
                db.commit()

                # Log false positive marking
                audit_log = AuditLog(
                    user_id=marked_by,
                    action="alert_marked_false_positive",
                    resource_type="alert",
                    resource_id=alert_uuid,
                    action_details={
                        "marked_at": datetime.now().isoformat(),
                        "notes": notes
                    },
                    success=True
                )
                db.add(audit_log)
                db.commit()

                db.close()
                return True

            db.close()
            return False

        except Exception as e:
            logger.error(f"Error marking false positive: {e}")
            return False
