#!/usr/bin/env python3
"""
IoT Attack Emulation Script for Network Threat Detection Testbed
Generates both benign and malicious IoT traffic for model validation.
"""

import json
import random
import time
import threading
import logging
import argparse
import boto3
from datetime import datetime
from typing import Dict, List, Optional
import paho.mqtt.client as mqtt
import ssl
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IoTDevice:
    """Represents a simulated IoT device that can generate benign or malicious traffic."""
    
    def __init__(self, device_id: str, device_type: str, 
                 mqtt_client: mqtt.Client, topic_prefix: str = "device"):
        self.device_id = device_id
        self.device_type = device_type
        self.mqtt_client = mqtt_client
        self.topic = f"{topic_prefix}/{device_id}/data"
        self.is_compromised = False
        self.attack_type = None
        self.running = True
        
        # Device-specific parameters
        self.base_temp = random.uniform(20.0, 26.0)
        self.temp_variance = random.uniform(0.5, 2.0)
        self.normal_interval = random.uniform(30, 120)  # seconds
        
    def generate_benign_payload(self) -> Dict:
        """Generate normal IoT device telemetry data."""
        # Temperature with realistic variations
        temp = self.base_temp + random.uniform(-self.temp_variance, self.temp_variance)
        
        # Realistic device status
        status = random.choice(['NORMAL', 'IDLE', 'ACTIVE'])
        
        # Battery level (if applicable)
        battery = random.uniform(85, 100) if self.device_type in ['sensor', 'tracker'] else None
        
        payload = {
            'timestamp': int(time.time()),
            'device_id': self.device_id,
            'device_type': self.device_type,
            'temperature': round(temp, 2),
            'status': status,
            'sequence_number': random.randint(1, 10000),
            'firmware_version': '1.2.3'
        }
        
        if battery:
            payload['battery_level'] = round(battery, 1)
            
        return payload
    
    def generate_malicious_payload(self, attack_type: str) -> Dict:
        """Generate malicious traffic based on attack type."""
        base_payload = self.generate_benign_payload()
        
        if attack_type == 'ddos_flood':
            # DDoS: High frequency, small payloads
            base_payload['status'] = 'ATTACK_DDOS'
            base_payload['attack_signature'] = 'ddos_flood'
            
        elif attack_type == 'data_exfiltration':
            # Data exfiltration: Large payloads with sensitive data
            base_payload['status'] = 'ATTACK_EXFIL'
            base_payload['attack_signature'] = 'data_exfiltration'
            # Simulate exfiltrated data
            base_payload['exfiltrated_data'] = 'x' * random.randint(1024, 5120)  # 1-5KB
            base_payload['sensitive_info'] = {
                'user_credentials': 'admin:password123',
                'network_config': '192.168.1.0/24',
                'device_keys': str(uuid.uuid4())
            }
            
        elif attack_type == 'c2_beacon':
            # C2 beacons: Small, periodic, with unusual values
            base_payload['status'] = 'ATTACK_C2'
            base_payload['attack_signature'] = 'c2_beacon'
            base_payload['temperature'] = -999  # Unusual value
            base_payload['c2_command'] = random.choice(['heartbeat', 'update', 'execute'])
            base_payload['c2_server'] = '192.168.99.100'
            
        elif attack_type == 'scanning':
            # Network scanning: Port scanning patterns
            base_payload['status'] = 'ATTACK_SCAN'
            base_payload['attack_signature'] = 'network_scanning'
            base_payload['scanned_ports'] = random.sample(range(1, 65535), 10)
            base_payload['target_network'] = '192.168.1.0/24'
            
        elif attack_type == 'backdoor':
            # Backdoor: Unauthorized access patterns
            base_payload['status'] = 'ATTACK_BACKDOOR'
            base_payload['attack_signature'] = 'backdoor_access'
            base_payload['unauthorized_access'] = True
            base_payload['backdoor_command'] = random.choice(['shell', 'file_access', 'privilege_escalation'])
            
        elif attack_type == 'injection':
            # Injection attacks: Malicious code injection
            base_payload['status'] = 'ATTACK_INJECTION'
            base_payload['attack_signature'] = 'code_injection'
            base_payload['injected_code'] = "'; DROP TABLE users; --"
            base_payload['injection_type'] = random.choice(['sql', 'command', 'script'])
            
        return base_payload
    
    def publish_message(self, payload: Dict) -> None:
        """Publish message to MQTT broker."""
        try:
            message = json.dumps(payload)
            result = self.mqtt_client.publish(self.topic, message, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published message from {self.device_id}: {payload['status']}")
            else:
                logger.error(f"Failed to publish message from {self.device_id}: {result.rc}")
                
        except Exception as e:
            logger.error(f"Error publishing message from {self.device_id}: {e}")
    
    def start_benign_behavior(self) -> None:
        """Start normal device behavior in a separate thread."""
        def benign_loop():
            while self.running:
                if not self.is_compromised:
                    payload = self.generate_benign_payload()
                    self.publish_message(payload)
                    time.sleep(self.normal_interval)
                else:
                    time.sleep(1)  # Short sleep when compromised
        
        thread = threading.Thread(target=benign_loop, daemon=True)
        thread.start()
        logger.info(f"Started benign behavior for device {self.device_id}")
    
    def start_attack_behavior(self, attack_type: str, duration: int = 60) -> None:
        """Start attack behavior for specified duration."""
        def attack_loop():
            self.is_compromised = True
            self.attack_type = attack_type
            start_time = time.time()
            
            logger.warning(f"Device {self.device_id} compromised - starting {attack_type} attack")
            
            while self.running and (time.time() - start_time) < duration:
                payload = self.generate_malicious_payload(attack_type)
                self.publish_message(payload)
                
                # Different attack patterns have different frequencies
                if attack_type == 'ddos_flood':
                    time.sleep(random.uniform(0.1, 0.5))  # High frequency
                elif attack_type == 'c2_beacon':
                    time.sleep(random.uniform(10, 30))  # Periodic beacons
                elif attack_type == 'scanning':
                    time.sleep(random.uniform(1, 3))  # Moderate frequency
                else:
                    time.sleep(random.uniform(5, 15))  # Low frequency
            
            # Return to normal behavior
            self.is_compromised = False
            self.attack_type = None
            logger.info(f"Device {self.device_id} attack behavior ended - returning to normal")
        
        thread = threading.Thread(target=attack_loop, daemon=True)
        thread.start()
    
    def stop(self) -> None:
        """Stop device simulation."""
        self.running = False
        logger.info(f"Stopped device {self.device_id}")


class AttackEmulator:
    """Main attack emulation orchestrator."""
    
    def __init__(self, iot_endpoint: str, num_devices: int = 10):
        self.iot_endpoint = iot_endpoint
        self.num_devices = num_devices
        self.devices: List[IoTDevice] = []
        self.mqtt_client = None
        self.running = False
        
        # Attack scenario configurations
        self.attack_scenarios = [
            'ddos_flood',
            'data_exfiltration',
            'c2_beacon',
            'scanning',
            'backdoor',
            'injection'
        ]
        
        # Device types for realistic simulation
        self.device_types = [
            'thermostat',
            'sensor',
            'camera',
            'tracker',
            'gateway',
            'actuator'
        ]
    
    def setup_mqtt_client(self) -> None:
        """Setup MQTT client with AWS IoT Core."""
        self.mqtt_client = mqtt.Client()
        
        # Setup TLS
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        self.mqtt_client.tls_set_context(context)
        
        # Callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_publish = self.on_publish
        
        try:
            logger.info(f"Connecting to AWS IoT Core: {self.iot_endpoint}")
            self.mqtt_client.connect(self.iot_endpoint, 8883, 60)
            self.mqtt_client.loop_start()
            
        except Exception as e:
            logger.error(f"Failed to connect to AWS IoT Core: {e}")
            raise
    
    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("Connected to AWS IoT Core successfully")
        else:
            logger.error(f"Failed to connect to AWS IoT Core: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        logger.warning(f"Disconnected from AWS IoT Core: {rc}")
    
    def on_publish(self, client, userdata, mid):
        """MQTT publish callback."""
        logger.debug(f"Message published: {mid}")
    
    def create_devices(self) -> None:
        """Create simulated IoT devices."""
        for i in range(self.num_devices):
            device_id = f"device_{i:03d}"
            device_type = random.choice(self.device_types)
            
            device = IoTDevice(
                device_id=device_id,
                device_type=device_type,
                mqtt_client=self.mqtt_client
            )
            
            self.devices.append(device)
            logger.info(f"Created device {device_id} of type {device_type}")
    
    def start_benign_simulation(self) -> None:
        """Start benign traffic simulation for all devices."""
        logger.info("Starting benign traffic simulation...")
        
        for device in self.devices:
            device.start_benign_behavior()
        
        logger.info(f"Started benign simulation for {len(self.devices)} devices")
    
    def launch_attack_scenario(self, attack_type: str, 
                             num_compromised: int = None,
                             duration: int = 60) -> None:
        """Launch a specific attack scenario."""
        if num_compromised is None:
            num_compromised = random.randint(1, max(1, len(self.devices) // 3))
        
        # Select random devices to compromise
        compromised_devices = random.sample(self.devices, num_compromised)
        
        logger.warning(f"Launching {attack_type} attack on {num_compromised} devices for {duration} seconds")
        
        for device in compromised_devices:
            device.start_attack_behavior(attack_type, duration)
        
        return compromised_devices
    
    def run_continuous_simulation(self, attack_frequency: int = 300) -> None:
        """Run continuous simulation with periodic attacks."""
        logger.info("Starting continuous attack simulation...")
        self.running = True
        
        # Start benign simulation
        self.start_benign_simulation()
        
        # Periodic attack simulation
        while self.running:
            try:
                # Random interval between attacks
                sleep_time = random.uniform(attack_frequency * 0.5, attack_frequency * 1.5)
                logger.info(f"Waiting {sleep_time:.1f} seconds until next attack scenario...")
                time.sleep(sleep_time)
                
                if not self.running:
                    break
                
                # Random attack scenario
                attack_type = random.choice(self.attack_scenarios)
                num_compromised = random.randint(1, max(1, len(self.devices) // 4))
                duration = random.randint(30, 120)
                
                self.launch_attack_scenario(attack_type, num_compromised, duration)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping simulation...")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in continuous simulation: {e}")
                time.sleep(30)
    
    def stop(self) -> None:
        """Stop attack emulation."""
        logger.info("Stopping attack emulation...")
        self.running = False
        
        # Stop all devices
        for device in self.devices:
            device.stop()
        
        # Disconnect MQTT client
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        logger.info("Attack emulation stopped")


def main():
    """Main function to run attack emulation."""
    parser = argparse.ArgumentParser(description='IoT Attack Emulation for Threat Detection')
    parser.add_argument('--endpoint', required=True, help='AWS IoT Core endpoint')
    parser.add_argument('--devices', type=int, default=10, help='Number of simulated devices')
    parser.add_argument('--attack-frequency', type=int, default=300, 
                       help='Average seconds between attack scenarios')
    parser.add_argument('--mode', choices=['continuous', 'single'], default='continuous',
                       help='Simulation mode')
    parser.add_argument('--attack-type', choices=[
        'ddos_flood', 'data_exfiltration', 'c2_beacon', 
        'scanning', 'backdoor', 'injection'
    ], help='Single attack type (for single mode)')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Attack duration in seconds (for single mode)')
    
    args = parser.parse_args()
    
    # Create attack emulator
    emulator = AttackEmulator(
        iot_endpoint=args.endpoint,
        num_devices=args.devices
    )
    
    try:
        # Setup MQTT connection
        emulator.setup_mqtt_client()
        time.sleep(2)  # Allow connection to establish
        
        # Create devices
        emulator.create_devices()
        
        if args.mode == 'continuous':
            # Run continuous simulation
            emulator.run_continuous_simulation(args.attack_frequency)
        else:
            # Run single attack scenario
            if not args.attack_type:
                args.attack_type = random.choice(emulator.attack_scenarios)
            
            emulator.start_benign_simulation()
            time.sleep(10)  # Allow benign traffic to establish
            
            logger.info(f"Running single attack scenario: {args.attack_type}")
            emulator.launch_attack_scenario(args.attack_type, duration=args.duration)
            
            # Keep running for attack duration + buffer
            time.sleep(args.duration + 30)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
    finally:
        emulator.stop()


if __name__ == "__main__":
    main()