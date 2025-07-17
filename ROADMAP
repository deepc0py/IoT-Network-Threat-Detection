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

* **Ticket ID:** IOT-R-07 (New)
    * **Task:** **(SPIKE) AWS Testbed Design.** Research how to set up a cost-effective AWS cluster to emulate IoT network traffic for our live test. The primary goal is to design a configuration that stays within the **AWS free tier** while still providing a realistic environment for model validation.
    * **Assignee:** To be assigned.

---

### **Phase 2: Core Modeling and Evaluation (Due: July 25th)**

This phase focuses on selecting and building the best possible model within the time available.

* **Ticket ID:** IOT-R-08 (New)
    * **Task:** **(SPIKE) Optimal Model Selection.** Conduct a technical investigation to determine the best-performing model type for this specific problem and dataset. [cite_start]The **Random Forest classifier** will be the starting point[cite: 22], but this SPIKE should quickly evaluate its suitability against other potential candidates (e.g., Gradient Boosting, SVM).
    * **Assignee:** To be assigned.

* **Ticket ID:** IOT-R-03
    * **Task:** **Baseline Model Training & Tuning.** Based on the outcome of the model SPIKE (IOT-R-08), develop and train the chosen classifier. Train both binary and multi-class versions and perform hyperparameter tuning to optimize performance.
    * **Assignee:** To be assigned.

* **Ticket ID:** IOT-R-04
    * [cite_start]**Task:** **Model Evaluation Framework.** Build a reusable script to evaluate model performance using **precision, recall, and F1-score**, as specified in the project baseline[cite: 40].
    * **Assignee:** To be assigned.

---

### **Phase 3: Finalization and Delivery (Due: July 29th)**

The final days are dedicated to practical validation, compiling results, and creating the final deliverables.

* **Ticket ID:** IOT-R-09 (New)
    * **Task:** **Create AWS Test Cluster.** Following the design from the SPIKE ticket (IOT-R-07), build the AWS cluster. [cite_start]This environment will be used for a live network test to confirm the model's functionality in a simulated real-world scenario[cite: 40].
    * **Assignee:** To be assigned.

* **Ticket ID:** IOT-R-05 (Stretch Goal)
    * [cite_start]**Task:** **Anomaly Detection Model.** If time permits, implement an anomaly detection method (e.g., Isolation Forest) to address the project goal of evaluating methods for detecting unseen threats[cite: 39].
    * **Assignee:** To be assigned.
