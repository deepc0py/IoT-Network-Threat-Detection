#!/bin/bash

# Setup script for IoT Attack Emulator on EC2 instance
# Installs Node.js, builds TypeScript code, and configures the environment

set -e

echo "Setting up IoT Attack Emulator environment..."

# Update system packages
echo "Updating system packages..."
sudo yum update -y

# Install Node.js and npm
echo "Installing Node.js and npm..."
curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
sudo yum install -y nodejs

# Verify installation
echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"

# Install global TypeScript compiler
echo "Installing TypeScript globally..."
sudo npm install -g typescript ts-node

# Install project dependencies
echo "Installing project dependencies..."
npm install

# Build TypeScript code
echo "Building TypeScript code..."
npm run build

# Make the compiled JavaScript executable
chmod +x dist/iot-attack-emulator.js

# Create systemd service file for attack emulator
echo "Creating systemd service file..."
sudo tee /etc/systemd/system/iot-attack-emulator.service > /dev/null <<EOF
[Unit]
Description=IoT Attack Emulator Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/attack-scripts
ExecStart=/usr/bin/node dist/iot-attack-emulator.js --endpoint \${IOT_ENDPOINT} --devices 10 --mode continuous
Restart=always
RestartSec=10
Environment=NODE_ENV=production
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Create log directory
sudo mkdir -p /var/log/iot-attack-emulator
sudo chown ec2-user:ec2-user /var/log/iot-attack-emulator

# Create configuration file
echo "Creating configuration file..."
cat > config.json <<EOF
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
    },
    "c2_beacon": {
      "enabled": true,
      "frequency": 0.25,
      "duration": 180,
      "intensity": "low"
    },
    "scanning": {
      "enabled": true,
      "frequency": 0.2,
      "duration": 90,
      "intensity": "medium"
    },
    "backdoor": {
      "enabled": true,
      "frequency": 0.1,
      "duration": 150,
      "intensity": "high"
    },
    "injection": {
      "enabled": true,
      "frequency": 0.1,
      "duration": 75,
      "intensity": "medium"
    }
  }
}
EOF

# Create startup script
echo "Creating startup script..."
cat > start-emulator.sh <<EOF
#!/bin/bash

# IoT Attack Emulator Startup Script

# Check if IOT_ENDPOINT environment variable is set
if [ -z "\$IOT_ENDPOINT" ]; then
    echo "Error: IOT_ENDPOINT environment variable is not set"
    echo "Usage: IOT_ENDPOINT=your-iot-endpoint.amazonaws.com ./start-emulator.sh"
    exit 1
fi

echo "Starting IoT Attack Emulator..."
echo "IoT Endpoint: \$IOT_ENDPOINT"

# Start the emulator
node dist/iot-attack-emulator.js \\
    --endpoint "\$IOT_ENDPOINT" \\
    --devices 10 \\
    --mode continuous \\
    --attack-frequency 300 \\
    2>&1 | tee /var/log/iot-attack-emulator/emulator.log
EOF

chmod +x start-emulator.sh

# Create monitoring script
echo "Creating monitoring script..."
cat > monitor-emulator.sh <<EOF
#!/bin/bash

# IoT Attack Emulator Monitoring Script

echo "=== IoT Attack Emulator Status ==="
echo "Service Status:"
sudo systemctl status iot-attack-emulator --no-pager

echo ""
echo "Recent Logs:"
sudo journalctl -u iot-attack-emulator -n 20 --no-pager

echo ""
echo "Log Files:"
ls -la /var/log/iot-attack-emulator/

echo ""
echo "Network Connections:"
netstat -an | grep :8883

echo ""
echo "Process Information:"
ps aux | grep iot-attack-emulator
EOF

chmod +x monitor-emulator.sh

# Create stop script
echo "Creating stop script..."
cat > stop-emulator.sh <<EOF
#!/bin/bash

# IoT Attack Emulator Stop Script

echo "Stopping IoT Attack Emulator..."

# Stop systemd service
sudo systemctl stop iot-attack-emulator

# Kill any remaining processes
pkill -f "iot-attack-emulator"

echo "IoT Attack Emulator stopped."
EOF

chmod +x stop-emulator.sh

echo "Setup completed successfully!"
echo ""
echo "Usage:"
echo "  1. Set IoT endpoint: export IOT_ENDPOINT=your-endpoint.amazonaws.com"
echo "  2. Start emulator:   ./start-emulator.sh"
echo "  3. Monitor:          ./monitor-emulator.sh"
echo "  4. Stop emulator:    ./stop-emulator.sh"
echo ""
echo "Or use systemd service:"
echo "  sudo systemctl start iot-attack-emulator"
echo "  sudo systemctl enable iot-attack-emulator  # Start on boot"
echo "  sudo systemctl stop iot-attack-emulator"
echo ""
echo "View logs:"
echo "  tail -f /var/log/iot-attack-emulator/emulator.log"
echo "  sudo journalctl -u iot-attack-emulator -f"