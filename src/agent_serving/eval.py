# Databricks notebook source

# COMMAND ----------
%uv pip install "mlflow[databricks]" "databricks-langchain"
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
from mlflow.genai.scorers import Safety, RelevanceToQuery, Guidelines
from mlflow.genai.scorers import scorer

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/hr-ops-agent")

# COMMAND ----------
# SECTION 1: Eval dataset
# These are your test cases — question + what the correct answer should look like
# Grow this over time from real production traces

eval_data = [
    {
        "inputs": {
            "input": [{"role": "user", "content": "How many leave days does EMP001 have?"}]
        },
        "expected_response": "EMP001 has 12 remaining leave days."
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "What is the headcount in Engineering?"}]
        },
        "expected_response": "The Engineering department has 42 employees."
    },
    {
        "inputs": {
            "input": [{"role": "user", "content": "How many leave days does EMP002 have?"}]
        },
        "expected_response": "EMP002 has 5 remaining leave days."
    },
]

# COMMAND ----------
# SECTION 2: Built-in LLM judges
# These are pre-built scorers Databricks has already tuned
# Each one uses an LLM to grade your agent's output

# Safety     — flags harmful, toxic, or inappropriate content
# RelevanceToQuery — is the answer actually on topic
# Guidelines — your own plain English rules the judge checks against

my_guidelines = Guidelines(
    name="hr_conduct",
    guidelines=[
        "Always state the exact number — never say 'some' or 'a few'.",
        "Always mention the employee ID or department name in the response.",
        "Never reveal information about employees other than what was asked.",
        "Keep responses under 50 words.",
    ]
)

scorers = [
    Safety(),
    RelevanceToQuery(),
    my_guidelines,
]

# COMMAND ----------
# SECTION 3: Custom code scorer
# This is deterministic — no LLM involved
# Use for simple checks you can verify with Python

@scorer
def response_is_concise(outputs: str, **kwargs) -> bool:
    """Checks the response is under 50 words."""
    return len(outputs.split()) <= 50

@scorer
def mentions_number(outputs: str, **kwargs) -> bool:
    """Checks the response contains at least one number."""
    return any(char.isdigit() for char in outputs)

# add custom scorers to the list
all_scorers = scorers + [response_is_concise, mentions_number]

# COMMAND ----------
# SECTION 4: The predict function
# This is what mlflow.genai.evaluate calls for each row
# It calls your deployed endpoint directly

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")

def predict_fn(inputs):
    response = deploy_client.predict(
        endpoint="dev_dattada_vijay_hr-ops-agent-endpoint",
        inputs=inputs,
    )
    # extract the text from the response
    return response.output[0].content[0].text

# COMMAND ----------
# SECTION 5: Run evaluation
# mlflow.genai.evaluate runs every row through predict_fn
# then runs every scorer on the output
# logs everything to the MLflow experiment

with mlflow.start_run(run_name="hr_agent_eval_v1"):

    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=lambda inputs: predict_fn(inputs),
        scorers=all_scorers,
    )

    # log aggregate metrics
    mlflow.log_metric("avg_safety",      results.metrics.get("safety/mean", 0))
    mlflow.log_metric("avg_relevance",   results.metrics.get("relevance_to_query/mean", 0))
    mlflow.log_metric("avg_guidelines",  results.metrics.get("hr_conduct/mean", 0))
    mlflow.log_metric("avg_concise",     results.metrics.get("response_is_concise/mean", 0))
    mlflow.log_metric("avg_has_number",  results.metrics.get("mentions_number/mean", 0))

    print(results.metrics)