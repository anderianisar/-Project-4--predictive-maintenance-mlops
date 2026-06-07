# -Project-4--predictive-maintenance-mlops
# Predictive Maintenance MLOps Pipeline ⚙️📊

A production-ready Machine Learning Operations (MLOps) pipeline designed to predict industrial equipment failure. This repository contains the implementation of a full model lifecycle workflow, divided into experiment tracking (Lab 13) and model registry deployment with rollback capabilities (Lab 14) using **MLflow**, **XGBoost**, and **Scikit-Learn**.

---

## 🚀 Project Overview

The objective of this project is to build a systematic, reproducible machine learning infrastructure for predictive maintenance. Instead of treating machine learning as a series of isolated scripts, this project demonstrates how to organize ML code, log telemetry metrics automatically, version models, and manage real-world deployment states (Staging ➔ Production).

### Key Architectural Phases:
1. **Synthetic Data Engineering & EDA:** Simulating 10,000 realistic equipment sensor profiles (temperature, vibration, pressure, RPM, and maintenance age).
2. **Experiment Tracking (Lab 13):** Evaluating multiple algorithms (Logistic Regression, Random Forest, and XGBoost) while systematically logging parameters and performance indicators to an MLflow Tracking Server.
3. **Model Registry & Lifecycle Governance (Lab 14):** Isolating the champion model, adding metadata tags, promoting it through lifecycle environments, and verifying an automated fallback mechanism.

---

## 📊 Phase 1: Experiment Tracking (Lab 13)

In this phase, a tracking server was initialized to compare a collection of baseline classifier models. 

### Performance Metrics Captured
The pipeline automatically computes and streams the following metrics to the dashboard for evaluation:
* **Accuracy & Precision**
* **Recall & F1-Score**
* **ROC AUC (Area Under the Receiver Operating Characteristic)**

### Expected Baseline Performance
The algorithms converge to the following relative evaluation bands:
* **XGBoost:** Highest performance (`ROC AUC ~0.92 - 0.95`)
* **Random Forest:** Moderate performance (`ROC AUC ~0.90 - 0.93`)
* **Logistic Regression:** Linear baseline constraints (`ROC AUC ~0.85 - 0.88`)

---

## 📑 Phase 2: Model Registry & Governance (Lab 14)

Once the experiments were logged, the MLflow Model Registry was utilized to transition the optimal model into an operational asset.

### Workflow Implemented:
* **Setup Registry Client:** Connecting a programmatic client to query backend run databases and sort runs dynamically by performance.
* **Model Registration:** Storing the champion XGBoost run as a version-controlled entity (`PredictiveMaintenance` Version 1).
* **Metadata & Documentation Auditing:** Stamping the registered artifact with explicit tracking tags (`validation_status: passed`, `framework: xgboost`, `team: data-science`).
* **Lifecycle State Transitions:** Promoting the model through gated lifecycle environments:
  $$\text{Unassigned} \longrightarrow \text{Staging} \longrightarrow \text{Production}$$
* **Production Inference Pipeline:** Building an abstract operational wrapper function (`predict_equipment_failure`) that loads the current model using its active deployment stage tag, decoupling infrastructure modifications from runtime code.
* **Failover Rollback Testing:** Registering a secondary model version (Version 2) to simulate a faulty release, followed by an instantaneous rollback execution targeting the stable Version 1 asset.

---

## 🛠️ Tech Stack & Environment Setup

### Core Libraries Used
* **MLflow** (Experiment Tracking, Registry Client, PyFunc Loading)
* **XGBoost** (Gradient Boosting Classifier)
* **Scikit-Learn** (Data Splitting, Feature Scaling, Metrics Evaluation)
* **Pandas & NumPy** (Data Manipulation & Synthesis)
* **Matplotlib & Seaborn** (Exploratory Visualizations)

### Running Locally (PyCharm Setup)
1. Clone the repository and navigate to your project directory.
2. Initialize and activate your virtual environment (`.venv`).
3. Install the project dependencies:
   ```bash
   pip install mlflow scikit-learn pandas numpy matplotlib seaborn xgboost
