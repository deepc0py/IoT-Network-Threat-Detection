#!/usr/bin/env node
/**
 * IoT Attack Emulation Script for Network Threat Detection Testbed
 * TypeScript implementation optimized for network operations
 */

import * as mqtt from 'mqtt';
import * as AWS from 'aws-sdk';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { v4 as uuidv4 } from 'uuid';

interface DeviceConfig {
  deviceId: string;
  deviceType: string;
  baseTemp: number;
  tempVariance: number;
  normalInterval: number;
  isCompromised: boolean;
  attackType?: AttackType;
}

interface AttackPayload {
  timestamp: number;
  deviceId: string;
  deviceType: string;
  temperature: number;
  status: string;
  sequenceNumber: number;
  firmwareVersion: string;
  batteryLevel?: number;
  attackSignature?: string;
  [key: string]: any;
}

type AttackType = 'ddos_flood' | 'data_exfiltration' | 'c2_beacon' | 'scanning' | 'backdoor' | 'injection';

type DeviceType = 'thermostat' | 'sensor' | 'camera' | 'tracker' | 'gateway' | 'actuator';

class IoTDevice {
  private config: DeviceConfig;
  private mqttClient: mqtt.MqttClient;
  private topic: string;
  private running: boolean = false;
  private attackTimer?: NodeJS.Timeout;
  private normalTimer?: NodeJS.Timeout;
  private sequenceNumber: number = 0;

  constructor(
    deviceId: string,
    deviceType: DeviceType,
    mqttClient: mqtt.MqttClient,
    topicPrefix: string = 'device'
  ) {
    this.config = {
      deviceId,
      deviceType,
      baseTemp: 20 + Math.random() * 6, // 20-26°C
      tempVariance: 0.5 + Math.random() * 1.5, // 0.5-2°C
      normalInterval: 30000 + Math.random() * 90000, // 30-120 seconds
      isCompromised: false
    };
    
    this.mqttClient = mqttClient;
    this.topic = `${topicPrefix}/${deviceId}/data`;
    
    console.log(`Created device ${deviceId} of type ${deviceType}`);
  }

  private generateBenignPayload(): AttackPayload {
    const temp = this.config.baseTemp + (Math.random() - 0.5) * 2 * this.config.tempVariance;
    const statuses = ['NORMAL', 'IDLE', 'ACTIVE'];
    const status = statuses[Math.floor(Math.random() * statuses.length)];
    
    const payload: AttackPayload = {
      timestamp: Date.now(),
      deviceId: this.config.deviceId,
      deviceType: this.config.deviceType,
      temperature: Math.round(temp * 100) / 100,
      status,
      sequenceNumber: ++this.sequenceNumber,
      firmwareVersion: '1.2.3'
    };

    // Add battery level for mobile devices
    if (['sensor', 'tracker'].includes(this.config.deviceType)) {
      payload.batteryLevel = Math.round((85 + Math.random() * 15) * 10) / 10;
    }

    return payload;
  }

  private generateMaliciousPayload(attackType: AttackType): AttackPayload {
    const basePayload = this.generateBenignPayload();
    
    switch (attackType) {
      case 'ddos_flood':
        return {
          ...basePayload,
          status: 'ATTACK_DDOS',
          attackSignature: 'ddos_flood',
          packetSize: Math.floor(Math.random() * 1024) + 64, // 64-1088 bytes
          targetEndpoint: '192.168.1.1',
          floodRate: Math.floor(Math.random() * 1000) + 100 // packets/sec
        };

      case 'data_exfiltration':
        const sensitiveData = {
          userCredentials: 'admin:password123',
          networkConfig: '192.168.1.0/24',
          deviceKeys: uuidv4(),
          wifiPasswords: ['HomeWiFi123', 'GuestNetwork456'],
          locationData: { lat: 40.7128, lng: -74.0060 }
        };
        
        return {
          ...basePayload,
          status: 'ATTACK_EXFIL',
          attackSignature: 'data_exfiltration',
          exfiltratedData: 'x'.repeat(1024 + Math.floor(Math.random() * 4096)), // 1-5KB
          sensitiveInfo: sensitiveData,
          exfilMethod: Math.random() > 0.5 ? 'dns_tunneling' : 'http_post',
          encryptionUsed: Math.random() > 0.3
        };

      case 'c2_beacon':
        const commands = ['heartbeat', 'update', 'execute', 'download', 'upload'];
        return {
          ...basePayload,
          status: 'ATTACK_C2',
          attackSignature: 'c2_beacon',
          temperature: -999, // Unusual value
          c2Command: commands[Math.floor(Math.random() * commands.length)],
          c2Server: '192.168.99.100',
          beaconInterval: Math.floor(Math.random() * 300) + 60, // 1-5 minutes
          payload: crypto.randomBytes(16).toString('hex')
        };

      case 'scanning':
        const commonPorts = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 1433, 3389];
        const scannedPorts = Array.from({ length: 10 }, () => 
          Math.floor(Math.random() * 65535) + 1
        );
        
        return {
          ...basePayload,
          status: 'ATTACK_SCAN',
          attackSignature: 'network_scanning',
          scannedPorts: [...commonPorts, ...scannedPorts],
          targetNetwork: '192.168.1.0/24',
          scanType: Math.random() > 0.5 ? 'tcp_syn' : 'tcp_connect',
          portResponses: scannedPorts.map(port => ({
            port,
            status: Math.random() > 0.8 ? 'open' : 'closed'
          }))
        };

      case 'backdoor':
        const backdoorCommands = ['shell', 'file_access', 'privilege_escalation', 'persistence'];
        return {
          ...basePayload,
          status: 'ATTACK_BACKDOOR',
          attackSignature: 'backdoor_access',
          unauthorizedAccess: true,
          backdoorCommand: backdoorCommands[Math.floor(Math.random() * backdoorCommands.length)],
          executionTime: Math.floor(Math.random() * 10000) + 1000,
          privilegeLevel: Math.random() > 0.5 ? 'admin' : 'user',
          persistenceMechanism: Math.random() > 0.5 ? 'service' : 'registry'
        };

      case 'injection':
        const injectionPayloads = [
          "'; DROP TABLE users; --",
          '<script>alert("XSS")</script>',
          '$(rm -rf /)',
          '../../../etc/passwd',
          'admin\' OR \'1\'=\'1'
        ];
        
        return {
          ...basePayload,
          status: 'ATTACK_INJECTION',
          attackSignature: 'code_injection',
          injectedCode: injectionPayloads[Math.floor(Math.random() * injectionPayloads.length)],
          injectionType: ['sql', 'xss', 'command', 'path_traversal', 'ldap'][Math.floor(Math.random() * 5)],
          targetField: ['username', 'password', 'search', 'filename'][Math.floor(Math.random() * 4)],
          bypassAttempt: Math.random() > 0.3
        };

      default:
        return basePayload;
    }
  }

  private async publishMessage(payload: AttackPayload): Promise<void> {
    return new Promise((resolve, reject) => {
      const message = JSON.stringify(payload, null, 2);
      
      this.mqttClient.publish(this.topic, message, { qos: 1 }, (error) => {
        if (error) {
          console.error(`Failed to publish message from ${this.config.deviceId}:`, error);
          reject(error);
        } else {
          console.log(`[${this.config.deviceId}] Published: ${payload.status}`);
          resolve();
        }
      });
    });
  }

  public startBenignBehavior(): void {
    if (this.running) return;
    
    this.running = true;
    console.log(`Starting benign behavior for device ${this.config.deviceId}`);
    
    const benignLoop = async () => {
      if (!this.running || this.config.isCompromised) return;
      
      try {
        const payload = this.generateBenignPayload();
        await this.publishMessage(payload);
      } catch (error) {
        console.error(`Error in benign loop for ${this.config.deviceId}:`, error);
      }
      
      if (this.running && !this.config.isCompromised) {
        this.normalTimer = setTimeout(benignLoop, this.config.normalInterval);
      }
    };
    
    benignLoop();
  }

  public async startAttackBehavior(attackType: AttackType, duration: number = 60000): Promise<void> {
    console.warn(`Device ${this.config.deviceId} compromised - starting ${attackType} attack`);
    
    this.config.isCompromised = true;
    this.config.attackType = attackType;
    
    // Clear normal timer
    if (this.normalTimer) {
      clearTimeout(this.normalTimer);
      this.normalTimer = undefined;
    }
    
    const attackLoop = async () => {
      if (!this.running || !this.config.isCompromised) return;
      
      try {
        const payload = this.generateMaliciousPayload(attackType);
        await this.publishMessage(payload);
      } catch (error) {
        console.error(`Error in attack loop for ${this.config.deviceId}:`, error);
      }
      
      if (this.running && this.config.isCompromised) {
        // Different attack frequencies
        let nextInterval: number;
        switch (attackType) {
          case 'ddos_flood':
            nextInterval = 100 + Math.random() * 400; // 100-500ms
            break;
          case 'c2_beacon':
            nextInterval = 10000 + Math.random() * 20000; // 10-30s
            break;
          case 'scanning':
            nextInterval = 1000 + Math.random() * 2000; // 1-3s
            break;
          default:
            nextInterval = 5000 + Math.random() * 10000; // 5-15s
        }
        
        this.attackTimer = setTimeout(attackLoop, nextInterval);
      }
    };
    
    // Start attack immediately
    attackLoop();
    
    // Schedule return to normal behavior
    setTimeout(() => {
      this.stopAttackBehavior();
    }, duration);
  }

  public stopAttackBehavior(): void {
    if (!this.config.isCompromised) return;
    
    console.log(`Device ${this.config.deviceId} attack behavior ended - returning to normal`);
    
    this.config.isCompromised = false;
    this.config.attackType = undefined;
    
    // Clear attack timer
    if (this.attackTimer) {
      clearTimeout(this.attackTimer);
      this.attackTimer = undefined;
    }
    
    // Resume normal behavior
    if (this.running) {
      this.startBenignBehavior();
    }
  }

  public stop(): void {
    console.log(`Stopping device ${this.config.deviceId}`);
    
    this.running = false;
    
    if (this.normalTimer) {
      clearTimeout(this.normalTimer);
      this.normalTimer = undefined;
    }
    
    if (this.attackTimer) {
      clearTimeout(this.attackTimer);
      this.attackTimer = undefined;
    }
  }
}

class AttackEmulator {
  private iotEndpoint: string;
  private numDevices: number;
  private devices: IoTDevice[] = [];
  private mqttClient?: mqtt.MqttClient;
  private running: boolean = false;
  
  private readonly attackScenarios: AttackType[] = [
    'ddos_flood',
    'data_exfiltration',
    'c2_beacon',
    'scanning',
    'backdoor',
    'injection'
  ];
  
  private readonly deviceTypes: DeviceType[] = [
    'thermostat',
    'sensor',
    'camera',
    'tracker',
    'gateway',
    'actuator'
  ];

  constructor(iotEndpoint: string, numDevices: number = 10) {
    this.iotEndpoint = iotEndpoint;
    this.numDevices = numDevices;
  }

  private setupMqttClient(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Connecting to AWS IoT Core: ${this.iotEndpoint}`);
      
      // Setup MQTT client with AWS IoT Core
      const clientOptions: mqtt.IClientOptions = {
        port: 8883,
        protocol: 'mqtts',
        rejectUnauthorized: false,
        reconnectPeriod: 5000,
        connectTimeout: 30000,
        keepalive: 60
      };

      this.mqttClient = mqtt.connect(`mqtts://${this.iotEndpoint}`, clientOptions);
      
      this.mqttClient.on('connect', () => {
        console.log('Connected to AWS IoT Core successfully');
        resolve();
      });
      
      this.mqttClient.on('error', (error) => {
        console.error('MQTT connection error:', error);
        reject(error);
      });
      
      this.mqttClient.on('disconnect', () => {
        console.warn('Disconnected from AWS IoT Core');
      });
      
      this.mqttClient.on('reconnect', () => {
        console.log('Reconnecting to AWS IoT Core...');
      });
    });
  }

  private createDevices(): void {
    console.log(`Creating ${this.numDevices} simulated IoT devices...`);
    
    for (let i = 0; i < this.numDevices; i++) {
      const deviceId = `device_${i.toString().padStart(3, '0')}`;
      const deviceType = this.deviceTypes[Math.floor(Math.random() * this.deviceTypes.length)];
      
      const device = new IoTDevice(deviceId, deviceType, this.mqttClient!);
      this.devices.push(device);
    }
    
    console.log(`Created ${this.devices.length} devices`);
  }

  private startBenignSimulation(): void {
    console.log('Starting benign traffic simulation...');
    
    this.devices.forEach(device => {
      device.startBenignBehavior();
    });
    
    console.log(`Started benign simulation for ${this.devices.length} devices`);
  }

  private async launchAttackScenario(
    attackType: AttackType,
    numCompromised?: number,
    duration: number = 60000
  ): Promise<IoTDevice[]> {
    if (!numCompromised) {
      numCompromised = Math.floor(Math.random() * Math.max(1, this.devices.length / 3)) + 1;
    }
    
    // Select random devices to compromise
    const shuffled = [...this.devices].sort(() => 0.5 - Math.random());
    const compromisedDevices = shuffled.slice(0, numCompromised);
    
    console.warn(`Launching ${attackType} attack on ${numCompromised} devices for ${duration}ms`);
    
    // Start attack behavior on compromised devices
    const attackPromises = compromisedDevices.map(device => 
      device.startAttackBehavior(attackType, duration)
    );
    
    await Promise.all(attackPromises);
    
    return compromisedDevices;
  }

  public async runContinuousSimulation(attackFrequency: number = 300000): Promise<void> {
    console.log('Starting continuous attack simulation...');
    this.running = true;
    
    // Start benign simulation
    this.startBenignSimulation();
    
    // Periodic attack simulation
    while (this.running) {
      try {
        // Random interval between attacks
        const sleepTime = attackFrequency * 0.5 + Math.random() * attackFrequency;
        console.log(`Waiting ${Math.round(sleepTime / 1000)}s until next attack scenario...`);
        
        await new Promise(resolve => setTimeout(resolve, sleepTime));
        
        if (!this.running) break;
        
        // Random attack scenario
        const attackType = this.attackScenarios[Math.floor(Math.random() * this.attackScenarios.length)];
        const numCompromised = Math.floor(Math.random() * Math.max(1, this.devices.length / 4)) + 1;
        const duration = 30000 + Math.random() * 90000; // 30-120 seconds
        
        await this.launchAttackScenario(attackType, numCompromised, duration);
        
      } catch (error) {
        console.error('Error in continuous simulation:', error);
        await new Promise(resolve => setTimeout(resolve, 30000));
      }
    }
  }

  public async runSingleAttackScenario(
    attackType: AttackType,
    duration: number = 60000
  ): Promise<void> {
    console.log(`Running single attack scenario: ${attackType}`);
    
    // Start benign simulation
    this.startBenignSimulation();
    
    // Allow benign traffic to establish
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    // Launch attack
    await this.launchAttackScenario(attackType, undefined, duration);
    
    // Keep running for duration + buffer
    await new Promise(resolve => setTimeout(resolve, duration + 30000));
  }

  public async initialize(): Promise<void> {
    try {
      await this.setupMqttClient();
      this.createDevices();
      console.log('Attack emulator initialized successfully');
    } catch (error) {
      console.error('Failed to initialize attack emulator:', error);
      throw error;
    }
  }

  public stop(): void {
    console.log('Stopping attack emulation...');
    this.running = false;
    
    // Stop all devices
    this.devices.forEach(device => device.stop());
    
    // Disconnect MQTT client
    if (this.mqttClient) {
      this.mqttClient.end();
    }
    
    console.log('Attack emulation stopped');
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  
  // Parse command line arguments
  const endpoint = getArg(args, '--endpoint');
  const numDevices = parseInt(getArg(args, '--devices', '10'));
  const attackFrequency = parseInt(getArg(args, '--attack-frequency', '300')) * 1000;
  const mode = getArg(args, '--mode', 'continuous');
  const attackType = getArg(args, '--attack-type') as AttackType;
  const duration = parseInt(getArg(args, '--duration', '60')) * 1000;
  
  if (!endpoint) {
    console.error('Error: --endpoint is required');
    process.exit(1);
  }
  
  const emulator = new AttackEmulator(endpoint, numDevices);
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nReceived interrupt signal, stopping simulation...');
    emulator.stop();
    process.exit(0);
  });
  
  process.on('SIGTERM', () => {
    console.log('\nReceived termination signal, stopping simulation...');
    emulator.stop();
    process.exit(0);
  });
  
  try {
    await emulator.initialize();
    
    if (mode === 'continuous') {
      await emulator.runContinuousSimulation(attackFrequency);
    } else {
      const selectedAttackType = attackType || 
        ['ddos_flood', 'data_exfiltration', 'c2_beacon', 'scanning', 'backdoor', 'injection'][
          Math.floor(Math.random() * 6)
        ] as AttackType;
      
      await emulator.runSingleAttackScenario(selectedAttackType, duration);
    }
    
  } catch (error) {
    console.error('Error in main execution:', error);
    process.exit(1);
  }
}

function getArg(args: string[], name: string, defaultValue?: string): string {
  const index = args.indexOf(name);
  if (index !== -1 && index + 1 < args.length) {
    return args[index + 1];
  }
  return defaultValue || '';
}

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { AttackEmulator, IoTDevice, AttackType, DeviceType };