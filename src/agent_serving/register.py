# Databricks notebook source

import mlflow
import os
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/hr-ops-agent")

CATALOG = "main"
SCHEMA  = "default"

# get this by opening agent.py notebook → File → Copy path
notebook_dir = os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
AGENT_PATH = f"/Workspace{notebook_dir}/agent.py"
print(f"Agent path: {AGENT_PATH}")

# COMMAND ----------
with mlflow.start_run(run_name="register_hr_ops_agent"):

    model_info = mlflow.pyfunc.log_model(
        name="agent",             
        python_model=AGENT_PATH, 
        pip_requirements=[
            "mlflow[databricks]",
            "databricks-langchain",
            "unitycatalog-langchain[databricks]",
            "databricks-sdk",
            "langgraph==1.2.6",
            "langchain-core",
            "langchain",
        ],
        registered_model_name=f"{CATALOG}.{SCHEMA}.hr_ops_agent",
    )

    print(f"Registered: {CATALOG}.{SCHEMA}.hr_ops_agent")
    print(f"Version:    {model_info.registered_model_version}")
    print(f"URI:        {model_info.model_uri}")
