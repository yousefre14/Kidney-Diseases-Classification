# Kidney Disease Classification 🩺 | MLflow + DVC + MLOps

[![CI/CD](https://github.com/your-username/Kidney-Disease-Classification-MLflow-DVC/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/Kidney-Disease-Classification-MLflow-DVC/actions)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/Data%20Versioning-DVC-orange)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/Cloud-AWS-FF9900)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Project Overview

This project demonstrates a **production-ready Kidney Disease Classification system** built with **Deep Learning (TensorFlow/Keras + Transfer Learning)** and enhanced by full **MLOps lifecycle automation**.  

It showcases **end-to-end ML system design**:
- Modular training pipeline with reusable components.  
- Experiment tracking using **MLflow**.  
- Data versioning & pipeline orchestration via **DVC**.  
- Continuous Integration & Deployment using **GitHub Actions**.  
- Cloud deployment on **AWS (EC2, ECR)** with **Dockerized containers**.  

💡 My contribution: I designed, built, and automated the **ML workflow, experiment tracking, cloud deployment, and CI/CD pipeline**, highlighting my expertise in **MLOps, Deep Learning, and Cloud Engineering**.

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow, Keras, Transfer Learning (pre-trained CNNs)  
- **MLOps:** MLflow, DVC, Dagshub  
- **Cloud & Deployment:** AWS (EC2, ECR), Docker  
- **Automation:** GitHub Actions (CI/CD)  
- **Languages & Configs:** Python, YAML-based configuration  
- **Project Design:** Modular pipeline architecture  

---

## 🔄 Workflow Explanation

1. **Configs & Params:**  
   - `config.yaml` → manages data paths, MLflow setup, model registry.  
   - `params.yaml` → defines hyperparameters (epochs, lr, batch size).  

2. **Entities:** Structured objects pass configs/params into components.  

3. **Components:** Modular scripts for ingestion, preprocessing, model training, evaluation.  

4. **Pipelines:** Orchestrates components into a full ML workflow.  

5. **main.py:** Executes the ML pipeline with a single command.  

6. **dvc.yaml:** Defines pipeline stages for reproducibility and data versioning.  

---

## 📊 MLflow Integration

- Tracks **experiments, metrics, and model artifacts**.  
- Centralized logging for reproducibility.  
- Model registry for versioned deployments.  
- Integrated with **Dagshub** for remote storage.  

Run MLflow UI locally:

```bash
mlflow ui


---
## 📂 Project Structure

```bash
Kidney-Disease-Classification-MLflow-DVC/
│── config/
│   ├── config.yaml         # Global configurations
│   ├── params.yaml         # Hyperparameters
│
│── src/
│   ├── components/         # Data ingestion, training, evaluation
│   ├── pipelines/          # Orchestrated ML pipelines
│   ├── entity/             # Data entities for configs/params
│   ├── utils/              # Reusable helper functions
│   ├── main.py             # Entry point to run pipeline
│
│── dvc.yaml                # DVC pipeline orchestration
│── requirements.txt        # Dependencies
│── Dockerfile              # Containerization
│── .github/workflows/      # GitHub Actions for CI/CD
│── README.md               # Project documentation

---


