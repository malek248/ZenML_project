# Financial Prediction Pipeline with ZenML

## Project Overview

This repository contains a financial prediction project implemented using **[ZenML](https://docs.zenml.io/)** — an MLOps framework that enables reproducible and modular machine learning pipelines.

The project replicates an existing notebook-based workflow, now structured into a pipeline with ZenML to enable better orchestration, artifact management, and run traceability.

## Pipeline Structure
The implemented pipeline consists of the following custom steps:
- **data_processing** – Load and prepare raw financial data
- **eda** – Perform basic exploratory analysis (optional visual outputs)
- **feature_engineering** – Generate relevant features for modeling
- **model_training** – Train a supervised machine learning model
- **evaluation** – Compute performance metrics

Each step receives inputs and produces outputs called **artifacts** serialized in and deserialized from the **artifact store** using a **materializer**.
