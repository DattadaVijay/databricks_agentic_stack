# Databricks notebook source

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    DatabricksFunctionClient,
    set_uc_function_client,
)

from langchain.agents import create_agent

# COMMAND ----------
# SECTION 1: MLflow setup
mlflow.langchain.autolog()
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/hr-ops-agent")

CATALOG      = "main"
SCHEMA       = "default"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# COMMAND ----------
# SECTION 2: UC Function tools
# DatabricksFunctionClient defaults to serverless execution
client = DatabricksFunctionClient()
set_uc_function_client(client)

toolkit = UCFunctionToolkit(function_names=[
    f"{CATALOG}.{SCHEMA}.get_employee_leave_balance",
    f"{CATALOG}.{SCHEMA}.get_department_headcount",
])
tools = toolkit.tools

# COMMAND ----------
# SECTION 3: Agent
# create_agent replaces create_react_agent
# system_prompt replaces state_modifier
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)

graph = create_agent(
    llm,
    tools,
    system_prompt=(
        "You are an HR Operations assistant. "
        "Use get_employee_leave_balance for one employee's leave. "
        "Use get_department_headcount for team size questions. "
        "Be concise. Never invent data."
    ),
)

# COMMAND ----------
# SECTION 4: ResponsesAgent wrapper
import uuid

class HROpsAgent(ResponsesAgent):

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:

        # convert input messages to dicts for LangGraph
        input_messages = [
            {"role": message.role, "content": message.content}
            for message in request.input
        ]

        # run the agent
        result = graph.invoke({"messages": input_messages})
        final  = result["messages"][-1].content

        # count tool calls
        tool_call_count = 0
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_count += 1

        mlflow.log_metric("tool_calls", tool_call_count)
        mlflow.log_metric("response_chars", len(final))

        # use the built-in helper — handles id, type, status, content format for you
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=final,
                    id=str(uuid.uuid4()),
                )
            ]
        )

agent = HROpsAgent()

# COMMAND ----------
# SECTION 5: Quick test before registering
from mlflow.types.responses import Message

response = agent.predict({
    "input": [{"role": "user", "content": "How many leave days does EMP001 have?"}]
})

# get the text out of the response
print(response.output[0].content)

# COMMAND ----------
# SECTION 6: Register to UC
mlflow.set_registry_uri("databricks-uc")
mlflow.end_run()


with mlflow.start_run(run_name="register"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent,
        pip_requirements=[
            "mlflow[databricks]>=3.1",
            "databricks-langchain",
            "unitycatalog-langchain[databricks]",
            "databricks-sdk",
            "langgraph==1.2.6",
            "langchain-core",
        ],
        registered_model_name=f"{CATALOG}.{SCHEMA}.hr_ops_agent",
    )
    print(f"Registered version: {model_info.registered_model_version}")