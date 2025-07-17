

# **SPIKE Report IOT-R-07 (Revised): Design for a Malicious IoT Traffic Emulation Testbed**

## **1\. Executive Summary**

This report outlines a revised technical design for an AWS-based testbed specifically for emulating **malicious IoT network traffic** (Ticket IOT-R-07). The primary goal is to create a realistic, dynamic environment to validate the performance of our chosen anomaly detection model against a variety of attack vectors. The architecture is designed to be cost-effective, leveraging serverless components and staying within the AWS Free Tier where possible, while providing a robust platform for generating and capturing both benign and malicious data streams.

The architecture consists of a **Control Plane** for traffic generation and a **Data Plane** for ingestion and storage:

* **Control Plane:** A single Amazon EC2 instance acts as the attack emulation engine. It will run custom scripts capable of simulating a fleet of IoT devices and generating various malicious traffic patterns, such as those seen in DDoS attacks, data exfiltration, and command-and-control (C2) communications.1  
* **Data Plane:** A serverless pipeline using AWS IoT Core, Amazon SQS, AWS Lambda, and Amazon S3 will ingest, buffer, and store the generated traffic. This ensures that all simulated data, both benign and malicious, is captured reliably for subsequent model validation and analysis.

This design provides a flexible and scalable foundation for rigorously testing our anomaly detection capabilities against realistic threat scenarios.

## **2\. Malicious Traffic Emulation Architecture**

The testbed is designed to simulate a variety of common IoT attack vectors. The data flow is engineered for scalability and cost-efficiency, ensuring we can generate a rich dataset for model validation.

### **2.1. System Diagram for Malicious Traffic Emulation**

The following diagram illustrates the architecture designed to generate, ingest, and store simulated malicious and benign IoT traffic.

Code snippet

graph TD  
    subgraph "Control Plane: Attack Emulation Host"  
        A\[EC2 t2.micro Instance\] \--\>|Python Scripts Emulating...| B(Benign & Malicious Devices);  
        B \--\> C{AWS IoT Core};  
    end

    subgraph "Attacker Logic (on EC2)"  
        D \--\> B;  
        E \--\> B;  
        F \--\> B;  
    end

    subgraph "Data Plane: Ingestion & Storage"  
        C \-- "MQTT (TLS)" \--\> G;  
        G \-- "Forwards All Traffic" \--\> H;  
        I \-- "Polls Queue" \--\> H;  
        I \-- "Writes Batches" \--\> J;  
    end

    subgraph "Analysis & Validation"  
        K(ML Anomaly Detection Model) \-- "Reads Data For Validation" \--\> J;  
    end

### **2.2. Architectural Components**

1. **Attack Emulation Host (EC2 t2.micro)**: This single instance is the core of the simulation. It will host Python scripts that instantiate multiple logical IoT devices. Crucially, these scripts will be programmed to simulate various attack behaviors based on known IoT threat intelligence.1  
2. **Attacker Logic**: The emulation scripts will contain modules to generate specific malicious traffic patterns:  
   * **DDoS Simulation**: Scripts will emulate botnet-like behavior, such as multiple devices sending a high volume of small packets (UDP/TCP floods) or crafted MQTT Publish floods to a target endpoint, mimicking attacks seen in the wild.1  
   * **Data Exfiltration Simulation**: Devices will be programmed to occasionally send unusually large data payloads, simulating the theft of sensitive information.3  
   * **Command & Control (C2) Simulation**: Devices will periodically send small, regular "beacon" messages to an external (simulated) C2 server, a common tactic for malware to maintain contact with its operator.1  
   * **Scanning/Probing**: Devices will simulate network scanning behavior, attempting to connect to multiple ports or IP addresses, which is a common reconnaissance technique.9  
3. **Data Ingestion and Storage (IoT Core, SQS, Lambda, S3)**: This serverless pipeline remains the most efficient way to capture the generated traffic.  
   * **AWS IoT Core** acts as the secure, scalable MQTT broker for all device traffic (both benign and malicious).  
   * An **IoT Rule** captures every message and forwards it to an **SQS queue**. This decouples ingestion from processing and enables batching.  
   * A scheduled **AWS Lambda** function polls the SQS queue, retrieves messages in batches, and writes them to an **S3 bucket**. This batching approach is essential to stay within the S3 free tier's request limits and creates a time-partitioned data lake ideal for analysis.

## **3\. Implementation of Malicious Traffic Generation**

The key to a successful testbed is the quality and realism of the simulated malicious traffic.

### **3.1. Emulation Script Enhancements**

The Python script on the EC2 host will be modular, allowing us to easily enable or disable different attack scenarios.

Python

\# Example Snippet for Malicious Payload Generation  
import json  
import random  
import time

def generate\_payload(device\_id, is\_malicious=False, attack\_type=None):  
    """Generates a JSON payload, either benign or malicious."""  
      
    payload \= {  
        "timestamp": int(time.time()),  
        "device\_id": device\_id,  
        "temperature": round(random.uniform(20.0, 26.0), 2),  
        "status": "NORMAL"  
    }

    if is\_malicious:  
        if attack\_type \== 'ddos\_flood':  
            \# For DDoS, the payload might be small, but frequency is high.  
            \# The attack logic is handled by the publishing loop.  
            payload\['status'\] \= "ATTACK\_DDoS"  
          
        elif attack\_type \== 'data\_exfiltration':  
            \# Simulate exfiltration with an abnormally large data field  
            payload\['status'\] \= "ATTACK\_EXFIL"  
            payload\['extra\_data'\] \= 'x' \* 1024 \* 5 \# 5KB of junk data  
              
        elif attack\_type \== 'c2\_beacon':  
            \# C2 beacons are often small and periodic  
            payload\['status'\] \= "ATTACK\_C2\_BEACON"  
            payload\['temperature'\] \= \-999 \# Use an unusual value  
      
    return json.dumps(payload)

\# \--- In the main device simulation loop \---  
\# A device could be randomly selected to become part of a botnet  
is\_compromised \= random.random() \< 0.1 \# 10% chance of being a bot

if is\_compromised:  
    \# This compromised device will now generate malicious traffic  
    \# e.g., enter a high-frequency loop for a DDoS attack  
    for \_ in range(100): \# High frequency burst  
        malicious\_payload \= generate\_payload(device\_id, is\_malicious=True, attack\_type='ddos\_flood')  
        client.publish(topic, malicious\_payload)  
        time.sleep(0.1) \# Rapid-fire messages  
else:  
    \# Publish normal traffic  
    normal\_payload \= generate\_payload(device\_id)  
    client.publish(topic, normal\_payload)  
    time.sleep(60) \# Normal interval

### **3.2. Leveraging Open-Source Attack Simulation Tools**

For more sophisticated and standardized attack emulation, the EC2 host can be configured to run established open-source tools. This allows for reproducible, CVE-based attack scenarios.

* **IoT-Flock**: This tool is ideal for our use case as it can generate both normal and malicious traffic for MQTT and CoAP protocols. It explicitly supports simulating attacks like **MQTT Publish Floods** (a DDoS variant) and **MQTT Packet Crafting Attacks**.7 We can install its dependencies on the EC2 instance and run its console mode, feeding it XML configurations that define our desired attack scenarios.  
* **IoTSecSim**: This is a more advanced simulation framework that models not only attackers but also defense mechanisms. It supports a wider range of attacker tactics, including **DDoS, PDoS (Permanent Denial-of-Service), and Data Exfiltration**.3 While more complex to set up, it provides a powerful engine for simulating evolving attack behaviors.  
* **General Penetration Testing Tools**: Tools like **Metasploit**, **Nmap**, and **ZAP** can also be scripted and run from the EC2 instance to generate network scanning, probing, and vulnerability exploitation traffic against our simulated IoT device endpoints.10

By combining custom scripts with these powerful open-source tools, we can generate a highly realistic and diverse dataset of malicious traffic to rigorously validate the performance and robustness of our anomaly detection model.

#### **Works cited**

1. What are IoT Attacks? Vectors Examples and Prevention. \- Wallarm, accessed July 16, 2025, [https://www.wallarm.com/what/iot-attack](https://www.wallarm.com/what/iot-attack)  
2. Understanding IoT Attacks: Safeguarding Devices and Networks \- Tata Communications, accessed July 16, 2025, [https://www.tatacommunications.com/knowledge-base/internet-of-things-attacks/](https://www.tatacommunications.com/knowledge-base/internet-of-things-attacks/)  
3. kokonnchee/IoTSecSim \- GitHub, accessed July 16, 2025, [https://github.com/kokonnchee/IoTSecSim](https://github.com/kokonnchee/IoTSecSim)  
4. What is an Attack Vector? 16 Critical Examples \- UpGuard, accessed July 16, 2025, [https://www.upguard.com/blog/attack-vector](https://www.upguard.com/blog/attack-vector)  
5. Anatomy of attacks on IoT systems: review of attacks, impacts and countermeasures, accessed July 16, 2025, [https://www.oaepublish.com/articles/jsss.2022.07](https://www.oaepublish.com/articles/jsss.2022.07)  
6. Common Cyber-Attacks in the IoT \- GlobalSign, accessed July 16, 2025, [https://www.globalsign.com/en/blog/common-cyber-attacks-in-the-iot](https://www.globalsign.com/en/blog/common-cyber-attacks-in-the-iot)  
7. ThingzDefense/IoT-Flock: IoT-Flock is an open-source tool ... \- GitHub, accessed July 16, 2025, [https://github.com/ThingzDefense/IoT-Flock](https://github.com/ThingzDefense/IoT-Flock)  
8. Modelling DDoS attack in IoT network in NetSim, accessed July 16, 2025, [https://support.tetcos.com/support/solutions/articles/14000141596-simulating-simple-ddos-attacks-in-an-iot-network-in-netsim](https://support.tetcos.com/support/solutions/articles/14000141596-simulating-simple-ddos-attacks-in-an-iot-network-in-netsim)  
9. Smart deep learning model for enhanced IoT intrusion detection \- PMC \- PubMed Central, accessed July 16, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12215491/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12215491/)  
10. 20 Essential Open Source Cyber Security Tools for 2025 \- StationX, accessed July 16, 2025, [https://www.stationx.net/open-source-cyber-security-tools/](https://www.stationx.net/open-source-cyber-security-tools/)