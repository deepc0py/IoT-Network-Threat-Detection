# AWS IoT Attack Emulation Testbed

## Overview

This AWS-based testbed generates realistic IoT network traffic patterns, including both benign and malicious attacks, to validate our trained machine learning models for IoT threat detection.

## Architecture

### Control Plane
- **EC2 t2.micro instance** - Runs TypeScript attack emulation scripts
- **Node.js-based attack emulator** - Simulates multiple IoT devices
- **IoT-Flock integration** - Professional attack simulation toolkit

### Data Plane
- **AWS IoT Core** - Secure MQTT broker for device communication
- **SQS Queue** - Buffers incoming IoT messages
- **Lambda Function** - Processes messages and stores in S3
- **S3 Bucket** - Data lake for captured traffic

### Attack Types Simulated
1. **DDoS Flood** - High-frequency small packet attacks
2. **Data Exfiltration** - Large payload transfers with sensitive data
3. **C2 Beacons** - Command & control communication patterns
4. **Network Scanning** - Port scanning and reconnaissance
5. **Backdoor Access** - Unauthorized access patterns
6. **Code Injection** - SQL injection and XSS attacks

## Quick Start

### Prerequisites
- AWS CLI configured with appropriate permissions
- Python 3.8+ and pip
- An existing EC2 key pair

### 1. Deploy Infrastructure

```bash
# Clone the repository
git clone <repo-url>
cd IoT-Network-Threat-Detection/aws

# Deploy the testbed
python deploy_testbed.py --key-pair your-key-pair-name --devices 10
```

### 2. Monitor Traffic

```bash
# SSH to the instance
ssh -i your-key-pair.pem ec2-user@<instance-ip>

# Monitor attack emulator
cd /home/ec2-user/attack-scripts
./monitor-emulator.sh

# Check logs
tail -f /var/log/iot-attack-emulator/emulator.log
```

### 3. Access Data

```bash
# View captured traffic in S3
aws s3 ls s3://iot-threat-detection-traffic-data-<account-id>/traffic_data/

# Download traffic data for analysis
aws s3 sync s3://iot-threat-detection-traffic-data-<account-id>/traffic_data/ ./traffic_data/
```

## Detailed Usage

### Manual Deployment

1. **Deploy CloudFormation Stack**
   ```bash
   aws cloudformation create-stack \
     --stack-name iot-threat-detection-testbed \
     --template-body file://cloudformation/iot-testbed.yaml \
     --parameters ParameterKey=KeyPairName,ParameterValue=your-key-pair \
     --capabilities CAPABILITY_NAMED_IAM
   ```

2. **Setup Attack Scripts**
   ```bash
   # Copy scripts to instance
   scp -r attack-scripts/ ec2-user@<instance-ip>:/home/ec2-user/
   
   # SSH and setup
   ssh -i your-key.pem ec2-user@<instance-ip>
   cd /home/ec2-user/attack-scripts
   chmod +x setup.sh && ./setup.sh
   ```

3. **Start Attack Simulation**
   ```bash
   # Set IoT endpoint
   export IOT_ENDPOINT=<your-account-id>.iot.us-east-1.amazonaws.com
   
   # Start continuous simulation
   ./start-emulator.sh
   
   # Or start single attack
   node dist/iot-attack-emulator.js --endpoint $IOT_ENDPOINT --mode single --attack-type ddos_flood
   ```

### Attack Emulator Options

```bash
# Continuous simulation (default)
node dist/iot-attack-emulator.js \
  --endpoint your-iot-endpoint.amazonaws.com \
  --devices 10 \
  --mode continuous \
  --attack-frequency 300

# Single attack scenario
node dist/iot-attack-emulator.js \
  --endpoint your-iot-endpoint.amazonaws.com \
  --devices 5 \
  --mode single \
  --attack-type data_exfiltration \
  --duration 120
```

## Model Integration

### Validating Trained Models

1. **Download Traffic Data**
   ```bash
   # Download latest traffic data
   aws s3 sync s3://your-bucket/traffic_data/ ./validation_data/
   ```

2. **Run Model Validation**
   ```bash
   cd /path/to/IoT-Network-Threat-Detection
   
   # Process validation data
   python src/preprocess.py \
     --input_path ./validation_data/combined_traffic.csv \
     --output_path ./validation_processed/
   
   # Evaluate model performance
   python src/evaluate.py \
     --model_path models/xgboost_iot.pkl \
     --test_data_path ./validation_processed/test_data.csv \
     --output_dir ./validation_results/
   ```

3. **Live Model Validation**
   ```python
   import joblib
   import pandas as pd
   from src.preprocess import IoTDataPreprocessor
   
   # Load trained model
   model_data = joblib.load('models/xgboost_iot.pkl')
   model = model_data['model']
   scaler = model_data['scaler']
   
   # Load and preprocess live data
   live_data = pd.read_csv('validation_data/latest_traffic.csv')
   preprocessor = IoTDataPreprocessor()
   processed_data = preprocessor.process_live_data(live_data)
   
   # Make predictions
   predictions = model.predict(processed_data)
   probabilities = model.predict_proba(processed_data)
   
   # Analyze results
   attack_detections = predictions.sum()
   print(f"Detected {attack_detections} attacks out of {len(predictions)} samples")
   ```

## Configuration

### Attack Emulator Configuration

Edit `attack-scripts/config.json`:

```json
{
  "emulator": {
    "numDevices": 10,
    "attackFrequency": 300,
    "logLevel": "info"
  },
  "attacks": {
    "ddos_flood": {
      "enabled": true,
      "frequency": 0.2,
      "duration": 60,
      "intensity": "high"
    },
    "data_exfiltration": {
      "enabled": true,
      "frequency": 0.15,
      "duration": 120,
      "intensity": "medium"
    }
  }
}
```

### AWS Resources Configuration

Modify `cloudformation/iot-testbed.yaml` to adjust:
- Instance types
- S3 bucket lifecycle policies
- Lambda function memory/timeout
- SQS queue settings

## Monitoring and Troubleshooting

### Key Metrics to Monitor

1. **Attack Emulator Health**
   ```bash
   # Check service status
   sudo systemctl status iot-attack-emulator
   
   # View recent logs
   sudo journalctl -u iot-attack-emulator -f
   ```

2. **AWS Resources**
   ```bash
   # Check SQS queue depth
   aws sqs get-queue-attributes \
     --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/iot-threat-detection-traffic-queue \
     --attribute-names ApproximateNumberOfMessages
   
   # Check Lambda function metrics
   aws logs describe-log-groups --log-group-name-prefix /aws/lambda/iot-threat-detection-traffic-processor
   ```

3. **Data Flow**
   ```bash
   # Check S3 bucket for new data
   aws s3 ls s3://your-bucket/traffic_data/ --recursive --human-readable
   
   # Monitor MQTT topics
   aws iot-data get-thing-shadow --thing-name device_001
   ```

### Common Issues

1. **Connection Issues**
   - Check security group rules
   - Verify IoT Core endpoint
   - Check certificate configuration

2. **Performance Issues**
   - Monitor EC2 instance CPU/memory
   - Check SQS queue depth
   - Review Lambda function timeout

3. **Data Issues**
   - Verify S3 bucket permissions
   - Check Lambda function logs
   - Monitor CloudWatch metrics

## Security Considerations

### Access Control
- Use IAM roles with minimal permissions
- Restrict security group access to specific IP ranges
- Enable CloudTrail for API logging

### Data Protection
- All data encrypted in transit (TLS)
- S3 bucket encryption enabled
- VPC endpoints for private communication

### Cost Optimization
- t2.micro instances (free tier eligible)
- S3 lifecycle policies for data retention
- Lambda pricing optimization

## Cleanup

```bash
# Stop attack emulator
ssh -i your-key.pem ec2-user@<instance-ip>
cd /home/ec2-user/attack-scripts
./stop-emulator.sh

# Delete CloudFormation stack
python deploy_testbed.py --cleanup

# Or manually
aws cloudformation delete-stack --stack-name iot-threat-detection-testbed
```

## API Reference

### Attack Emulator API

```typescript
// Start attack simulation
const emulator = new AttackEmulator(iotEndpoint, numDevices);
await emulator.initialize();

// Run continuous simulation
await emulator.runContinuousSimulation(300000); // 5 minutes

// Run single attack
await emulator.runSingleAttackScenario('ddos_flood', 60000); // 1 minute
```

### AWS Resources

- **IoT Core Topic**: `device/{device_id}/data`
- **SQS Queue**: `iot-threat-detection-traffic-queue`
- **S3 Bucket**: `iot-threat-detection-traffic-data-{account-id}`
- **Lambda Function**: `iot-threat-detection-traffic-processor`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test changes with the testbed
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.