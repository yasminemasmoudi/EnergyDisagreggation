# Deep Learning model for energy disaggregation using NILM (Non-Intrusive Load Monitoring) techniques & MLOps pipeline implementation

# Getting started
Prerequisites: 

 1. Python 
 3. Install Anaconda 

## Model training with MLflow:

 1. clone this repo: `git clone https://github.com/yasminemasmoudi/EnergyDisagreggationSFM`
 2. create conda environment: `conda env create --name env_name`
 3. activate conda environment: `conda activate env_name`

## Run MLflow tracking UI:
In the same repo directory, run `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 `
UI is accessible at http://localhost:5000/

## Run Prefect UI:
In the same repo directory, run `prefect orion start`
UI is accessible at http://localhost:4200/
