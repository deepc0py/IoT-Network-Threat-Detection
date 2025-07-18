#!/usr/bin/env python3
"""
AWS IoT Attack Emulation Testbed Deployment Script
Deploys and configures the complete testbed infrastructure.
"""

import boto3
import json
import time
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestbedDeployer:
    """Deploys and manages the IoT attack emulation testbed."""
    
    def __init__(self, project_name: str = 'iot-threat-detection',
                 region: str = 'us-east-1'):
        self.project_name = project_name
        self.region = region
        self.stack_name = f"{project_name}-testbed"
        
        # Initialize AWS clients
        self.session = boto3.Session(region_name=region)
        self.cloudformation = self.session.client('cloudformation')
        self.ec2 = self.session.client('ec2')
        self.iot = self.session.client('iot')
        self.s3 = self.session.client('s3')
        
        # Deployment paths
        self.script_dir = Path(__file__).parent
        self.cloudformation_template = self.script_dir / 'cloudformation' / 'iot-testbed.yaml'
        self.attack_scripts_dir = self.script_dir / 'attack-scripts'
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        # Check if CloudFormation template exists
        if not self.cloudformation_template.exists():
            logger.error(f"CloudFormation template not found: {self.cloudformation_template}")
            return False
        
        # Check if attack scripts exist
        if not self.attack_scripts_dir.exists():
            logger.error(f"Attack scripts directory not found: {self.attack_scripts_dir}")
            return False
        
        # Check AWS credentials
        try:
            sts = self.session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS Identity: {identity['Arn']}")
        except Exception as e:
            logger.error(f"AWS credentials not configured: {e}")
            return False
        
        # Check if key pair exists
        try:
            key_pairs = self.ec2.describe_key_pairs()
            if not key_pairs['KeyPairs']:
                logger.error("No EC2 key pairs found. Please create one first.")
                return False
            logger.info(f"Available key pairs: {[kp['KeyName'] for kp in key_pairs['KeyPairs']]}")
        except Exception as e:
            logger.error(f"Error checking key pairs: {e}")
            return False
        
        logger.info("All prerequisites checked successfully")
        return True
    
    def create_key_pair(self, key_name: str) -> str:
        """Create a new EC2 key pair."""
        try:
            logger.info(f"Creating EC2 key pair: {key_name}")
            
            response = self.ec2.create_key_pair(KeyName=key_name)
            private_key = response['KeyMaterial']
            
            # Save private key to file
            key_file = self.script_dir / f"{key_name}.pem"
            with open(key_file, 'w') as f:
                f.write(private_key)
            
            # Set proper permissions
            key_file.chmod(0o600)
            
            logger.info(f"Key pair created and saved to: {key_file}")
            return str(key_file)
            
        except Exception as e:
            logger.error(f"Error creating key pair: {e}")
            raise
    
    def deploy_infrastructure(self, key_pair_name: str, 
                            allowed_cidr: str = '0.0.0.0/0') -> Dict[str, Any]:
        """Deploy the CloudFormation stack."""
        logger.info(f"Deploying infrastructure stack: {self.stack_name}")
        
        try:
            # Read CloudFormation template
            with open(self.cloudformation_template, 'r') as f:
                template_body = f.read()
            
            # Deploy stack
            parameters = [
                {
                    'ParameterKey': 'ProjectName',
                    'ParameterValue': self.project_name
                },
                {
                    'ParameterKey': 'KeyPairName',
                    'ParameterValue': key_pair_name
                },
                {
                    'ParameterKey': 'AllowedCIDR',
                    'ParameterValue': allowed_cidr
                }
            ]
            
            response = self.cloudformation.create_stack(
                StackName=self.stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            logger.info(f"Stack creation initiated: {response['StackId']}")
            
            # Wait for stack to complete
            logger.info("Waiting for stack deployment to complete...")
            waiter = self.cloudformation.get_waiter('stack_create_complete')
            waiter.wait(
                StackName=self.stack_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )
            
            # Get stack outputs
            stack_info = self.cloudformation.describe_stacks(StackName=self.stack_name)
            outputs = {}
            for output in stack_info['Stacks'][0].get('Outputs', []):
                outputs[output['OutputKey']] = output['OutputValue']
            
            logger.info("Infrastructure deployment completed successfully")
            return outputs
            
        except Exception as e:
            logger.error(f"Error deploying infrastructure: {e}")
            raise
    
    def wait_for_instance_ready(self, instance_id: str, timeout: int = 600) -> bool:
        """Wait for EC2 instance to be ready."""
        logger.info(f"Waiting for instance {instance_id} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.ec2.describe_instances(InstanceIds=[instance_id])
                instance = response['Reservations'][0]['Instances'][0]
                state = instance['State']['Name']
                
                if state == 'running':
                    # Check if instance is ready for SSH
                    status_response = self.ec2.describe_instance_status(
                        InstanceIds=[instance_id]
                    )
                    
                    if status_response['InstanceStatuses']:
                        status = status_response['InstanceStatuses'][0]
                        if (status['InstanceStatus']['Status'] == 'ok' and
                            status['SystemStatus']['Status'] == 'ok'):
                            logger.info(f"Instance {instance_id} is ready")
                            return True
                
                logger.info(f"Instance state: {state}, waiting...")
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error checking instance status: {e}")
                time.sleep(30)
        
        logger.error(f"Timeout waiting for instance {instance_id} to be ready")
        return False
    
    def deploy_attack_scripts(self, instance_ip: str, key_file: str) -> bool:
        """Deploy attack scripts to EC2 instance."""
        logger.info(f"Deploying attack scripts to {instance_ip}")
        
        try:
            # Copy attack scripts to instance
            scp_command = [
                'scp',
                '-i', key_file,
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                '-r', str(self.attack_scripts_dir),
                f'ec2-user@{instance_ip}:/home/ec2-user/'
            ]
            
            logger.info("Copying attack scripts to instance...")
            result = subprocess.run(scp_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"SCP failed: {result.stderr}")
                return False
            
            # Run setup script to install Node.js and build TypeScript
            ssh_command = [
                'ssh',
                '-i', key_file,
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                f'ec2-user@{instance_ip}',
                'cd /home/ec2-user/attack-scripts && chmod +x setup.sh && ./setup.sh'
            ]
            
            logger.info("Setting up Node.js and building TypeScript attack emulator...")
            result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Setup failed: {result.stderr}")
                return False
            
            logger.info("Attack scripts deployed and built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying attack scripts: {e}")
            return False
    
    def setup_iot_certificates(self, instance_ip: str, key_file: str) -> bool:
        """Setup IoT certificates on the EC2 instance."""
        logger.info("Setting up IoT certificates...")
        
        try:
            # Create IoT policy
            policy_name = f"{self.project_name}-policy"
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "iot:Connect",
                            "iot:Publish",
                            "iot:Subscribe",
                            "iot:Receive"
                        ],
                        "Resource": "*"
                    }
                ]
            }
            
            try:
                self.iot.create_policy(
                    policyName=policy_name,
                    policyDocument=json.dumps(policy_document)
                )
                logger.info(f"Created IoT policy: {policy_name}")
            except self.iot.exceptions.ResourceAlreadyExistsException:
                logger.info(f"IoT policy already exists: {policy_name}")
            
            # Create IoT certificate
            cert_response = self.iot.create_keys_and_certificate(setAsActive=True)
            
            certificate_arn = cert_response['certificateArn']
            certificate_pem = cert_response['certificatePem']
            private_key_pem = cert_response['keyPair']['PrivateKey']
            
            # Attach policy to certificate
            self.iot.attach_policy(
                policyName=policy_name,
                target=certificate_arn
            )
            
            # Save certificates to temporary files
            cert_file = self.script_dir / 'device_cert.pem'
            key_file_path = self.script_dir / 'device_key.pem'
            
            with open(cert_file, 'w') as f:
                f.write(certificate_pem)
            
            with open(key_file_path, 'w') as f:
                f.write(private_key_pem)
            
            # Copy certificates to instance
            scp_command = [
                'scp',
                '-i', key_file,
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                str(cert_file),
                str(key_file_path),
                f'ec2-user@{instance_ip}:/home/ec2-user/'
            ]
            
            logger.info("Copying certificates to instance...")
            result = subprocess.run(scp_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Certificate copy failed: {result.stderr}")
                return False
            
            # Clean up temporary files
            cert_file.unlink()
            key_file_path.unlink()
            
            logger.info("IoT certificates setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up IoT certificates: {e}")
            return False
    
    def start_attack_simulation(self, instance_ip: str, key_file: str,
                              iot_endpoint: str, num_devices: int = 10) -> bool:
        """Start the attack simulation on the EC2 instance."""
        logger.info("Starting attack simulation...")
        
        try:
            # Start TypeScript attack emulator
            ssh_command = [
                'ssh',
                '-i', key_file,
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'UserKnownHostsFile=/dev/null',
                f'ec2-user@{instance_ip}',
                f'cd /home/ec2-user/attack-scripts && IOT_ENDPOINT={iot_endpoint} ./start-emulator.sh'
            ]
            
            logger.info("Starting TypeScript attack emulator...")
            result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Attack simulation started successfully")
                logger.info(f"SSH to instance: ssh -i {key_file} ec2-user@{instance_ip}")
                logger.info("Monitor emulator: cd /home/ec2-user/attack-scripts && ./monitor-emulator.sh")
                logger.info("Check logs: tail -f /var/log/iot-attack-emulator/emulator.log")
                return True
            else:
                logger.error(f"Failed to start simulation: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Error starting attack simulation: {e}")
            return False
    
    def get_stack_outputs(self) -> Dict[str, Any]:
        """Get outputs from the deployed stack."""
        try:
            stack_info = self.cloudformation.describe_stacks(StackName=self.stack_name)
            outputs = {}
            for output in stack_info['Stacks'][0].get('Outputs', []):
                outputs[output['OutputKey']] = output['OutputValue']
            return outputs
        except Exception as e:
            logger.error(f"Error getting stack outputs: {e}")
            return {}
    
    def cleanup(self) -> bool:
        """Clean up deployed resources."""
        logger.info("Cleaning up deployed resources...")
        
        try:
            # Delete CloudFormation stack
            self.cloudformation.delete_stack(StackName=self.stack_name)
            
            # Wait for stack deletion
            logger.info("Waiting for stack deletion to complete...")
            waiter = self.cloudformation.get_waiter('stack_delete_complete')
            waiter.wait(StackName=self.stack_name)
            
            logger.info("Cleanup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False


def main():
    """Main function to deploy the testbed."""
    parser = argparse.ArgumentParser(description='Deploy IoT Attack Emulation Testbed')
    parser.add_argument('--project-name', default='iot-threat-detection',
                       help='Project name for resource naming')
    parser.add_argument('--region', default='us-east-1',
                       help='AWS region for deployment')
    parser.add_argument('--key-pair', help='EC2 key pair name (will create if not exists)')
    parser.add_argument('--allowed-cidr', default='0.0.0.0/0',
                       help='CIDR block allowed to access the testbed')
    parser.add_argument('--devices', type=int, default=10,
                       help='Number of simulated IoT devices')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up deployed resources')
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = TestbedDeployer(
        project_name=args.project_name,
        region=args.region
    )
    
    try:
        if args.cleanup:
            # Clean up resources
            success = deployer.cleanup()
            sys.exit(0 if success else 1)
        
        # Check prerequisites
        if not deployer.check_prerequisites():
            logger.error("Prerequisites not met")
            sys.exit(1)
        
        # Create or use existing key pair
        if args.key_pair:
            key_pair_name = args.key_pair
            key_file = None
        else:
            key_pair_name = f"{args.project_name}-key"
            try:
                # Check if key pair exists
                deployer.ec2.describe_key_pairs(KeyNames=[key_pair_name])
                logger.info(f"Using existing key pair: {key_pair_name}")
                key_file = None
            except:
                key_file = deployer.create_key_pair(key_pair_name)
        
        # Deploy infrastructure
        outputs = deployer.deploy_infrastructure(key_pair_name, args.allowed_cidr)
        
        instance_id = outputs['AttackEmulationInstanceId']
        instance_ip = outputs['AttackEmulationInstancePublicIP']
        iot_endpoint = outputs['IoTEndpoint']
        
        logger.info(f"Infrastructure deployed successfully!")
        logger.info(f"Instance ID: {instance_id}")
        logger.info(f"Instance IP: {instance_ip}")
        logger.info(f"IoT Endpoint: {iot_endpoint}")
        
        # Wait for instance to be ready
        if not deployer.wait_for_instance_ready(instance_id):
            logger.error("Instance not ready, aborting")
            sys.exit(1)
        
        # Use provided key file or default location
        if not key_file:
            key_file = f"{key_pair_name}.pem"
        
        # Deploy attack scripts
        if not deployer.deploy_attack_scripts(instance_ip, key_file):
            logger.error("Failed to deploy attack scripts")
            sys.exit(1)
        
        # Setup IoT certificates
        if not deployer.setup_iot_certificates(instance_ip, key_file):
            logger.error("Failed to setup IoT certificates")
            sys.exit(1)
        
        # Start attack simulation
        if not deployer.start_attack_simulation(instance_ip, key_file, iot_endpoint, args.devices):
            logger.error("Failed to start attack simulation")
            sys.exit(1)
        
        logger.info("="*60)
        logger.info("TESTBED DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Instance IP: {instance_ip}")
        logger.info(f"SSH Command: ssh -i {key_file} ec2-user@{instance_ip}")
        logger.info(f"S3 Bucket: {outputs['TrafficDataBucketName']}")
        logger.info(f"SQS Queue: {outputs['TrafficQueueURL']}")
        logger.info("="*60)
        logger.info("Monitor traffic data in S3 bucket and validate with your trained model!")
        
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()