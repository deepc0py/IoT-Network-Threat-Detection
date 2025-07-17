### **Accelerated Project Roadmap: IoT Threat Detection**

**Deadline:** July 29th, 2025

This plan incorporates research (SPIKE) and infrastructure tasks into the compressed two-week sprint. The focus remains on delivering a functional model and a final report, with live testing as a key validation step.

---

### **Phase 1: Data Processing and Environment Setup (Due: July 21st)**

This phase now includes foundational research for our testing environment alongside data preparation.

* **Ticket ID:** IOT-R-01
    * **Task:** **Infrastructure & Data Pipeline.** Set up the shared repository and development environment. [cite_start]Download the primary TON\_IoT dataset, write ingestion scripts, and perform a rapid EDA to understand features and class distributions[cite: 12, 14].
    * **Assignees:** Team members to be assigned.

* **Ticket ID:** IOT-R-02
    * **Task:** **Feature Engineering & Preprocessing.** Based on EDA, select features and prepare the data. [cite_start]This includes normalizing data to address feature scaling and handling categorical features[cite: 33]. [cite_start]It must also address class imbalance using techniques like class weighting[cite: 29].
    * **Assignees:** Team members to be assigned.

---

### **Phase 2: Core Modeling and Evaluation (Due: July 25th)**

This phase focuses on selecting and building the best possible model within the time available.

* **Ticket ID:** IOT-R-03
    * **Task:** **Model Implementation & Tuning (RF & XGBoost).** Develop and train the **Random Forest classifier** as a baseline. Then, implement and tune an **XGBoost classifier** to achieve the highest possible performance for the final model.
    * **Assignee:** To be assigned.

* **Ticket ID:** IOT-R-04
    * [cite_start]**Task:** **Model Evaluation Framework.** Build a reusable script to evaluate model performance using **precision, recall, and F1-score**, as specified in the project baseline[cite: 40].
    * **Assignee:** To be assigned.

---

### **Phase 3: Finalization and Delivery (Due: July 29th)**

The final days are dedicated to practical validation, compiling results, and creating the final deliverables.

* **Ticket ID:** IOT-R-09
    * **Task:** **Build AWS Attack Emulation Testbed.** Build the AWS testbed using a **t2.micro EC2 instance** for the control plane and a serverless data plane with **IoT Core, SQS, Lambda, and S3**. Configure the environment to use tools like **IoT-Flock** for traffic generation.
    * **Assignee:** To be assigned.

* **Ticket ID:** IOT-R-10 (New)
    * **Task:** **Generate & Collect Malicious Traffic Data.** Execute attack emulation scripts on the AWS testbed to generate and collect a dataset containing both benign and varied malicious traffic (e.g., DDoS, data exfiltration).
    * **Assignee:** To be assigned.

* **Ticket ID:** IOT-R-05 (Stretch Goal)
    * [cite_start]**Task:** **Anomaly Detection Model.** If time permits, implement an anomaly detection method (e.g., Isolation Forest) to address the project goal of evaluating methods for detecting unseen threats[cite: 39].
    * **Assignee:** To be assigned.
